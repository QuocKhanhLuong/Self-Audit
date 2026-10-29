"""Attach verified image geometry to a frozen evaluation manifest, without masks."""
import copy
import json
from pathlib import Path
import numpy as np
from .compact_artifacts import digest, _require_sha256
from .volume.geometry import validate_nifti_correspondence


def attach_geometry(manifest, manifest_sha, sources, sources_sha, out):
    """Derive a geometry-only manifest; original images, masks and split stay pinned.

    sources is a pinned mapping id -> raw cine NIfTI path/hash. We recompute the
    canonical image-only normalization and require voxel correspondence before
    accepting its affine. Missing physical evidence is an error, never identity.
    """
    import nibabel as nib
    manifest, sources, out = Path(manifest).resolve(), Path(sources).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('output exists')
    if digest(manifest) != _require_sha256(manifest_sha,'manifest SHA') or digest(sources) != _require_sha256(sources_sha,'sources SHA'):
        raise ValueError('geometry input hash mismatch')
    m=json.loads(manifest.read_text())
    source_map=json.loads(sources.read_text())
    if m.get('schema') != 'self_audit.data.v2' or m.get('role') not in {'development','external_frozen'}:
        raise ValueError('only canonical frozen evaluation manifests may be enriched')
    if source_map.get('schema') != 'self_audit.compact_geometry_sources.v1':
        raise ValueError('unsupported geometry source schema')
    records=m.get('patients',[])
    ids=[r['id'] for r in records]
    if not ids or len(set(ids))!=len(ids) or set(source_map.get('patients',{}))!=set(ids):
        raise ValueError('exact patient coverage required for geometry sources')
    enriched=copy.deepcopy(m)
    for row in enriched['patients']:
        image=Path(row['image'])
        if not image.is_absolute(): image=manifest.parent/image
        if digest(image)!=row['image_sha256']:
            raise ValueError('prepared image hash mismatch')
        prepared=np.load(image,mmap_mode='r',allow_pickle=False)
        source=source_map['patients'][row['id']]
        if set(source)!={'path','sha256'}:
            raise ValueError('geometry source must contain image path/hash only')
        native=Path(source['path'])
        if not native.is_absolute(): native=sources.parent/native
        if digest(native)!=_require_sha256(source['sha256'],'raw image SHA'):
            raise ValueError('raw image hash mismatch')
        obj=nib.load(str(native))
        affine=np.asarray(obj.affine)
        correspondence=validate_nifti_correspondence(
            native,image_sha256=source['sha256'],expected_shape_tzyx=tuple(prepared.shape),
            expected_affine_xyz_to_world_mm=affine,expected_orientation=''.join(nib.aff2axcodes(affine)),
            expected_frame_mapping=list(range(prepared.shape[0])),prepared_image_tzyx=prepared,masks=None,
            allow_unknown_spatial_units=True)
        row.update(image=str(image.resolve()),affine=affine.tolist(),axes=['T','Z','Y','X'],
                   spatial_units=correspondence['units'],
                   geometry_source_image={'path':str(native.resolve()),'sha256':source['sha256']})
        # Resolve paths for the new location, without opening reference labels.
        for mask in row.get('masks',{}).values():
            mp=Path(mask['path'])
            if not mp.is_absolute(): mask['path']=str((manifest.parent/mp).resolve())
    enriched['geometry_provenance']={'parent_manifest_sha256':manifest_sha,
                                     'raw_sources_manifest_sha256':sources_sha,
                                     'correspondence':'canonical percentile-normalized XYZT->TZYX image voxels',
                                     'spatial_units':'mm' if all(r['spatial_units']=='mm' for r in enriched['patients']) else 'unknown_in_source',
                                     'reference_masks_opened':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(enriched,indent=2,allow_nan=False)+'\n')
    return {'manifest':str(out),'manifest_sha256':digest(out),'parent_manifest_sha256':manifest_sha,
            'patients':len(ids),'reference_masks_opened':False}
