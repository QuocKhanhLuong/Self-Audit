import pytest
from scripts.build_teacher_memory_parallel import parser, validate_args


def test_parallel_a5_accepts_only_frozen_student_inputs():
    base=['--mode','a5','--out','new-output','--native-manifest','train.json',
          '--native-manifest-sha256','0'*64,'--student-checkpoint','A0/final',
          '--student-checkpoint-manifest-sha256','1'*64]
    validate_args(parser().parse_args(base))
    for forbidden in (['--teacher-weights','external.pt'],['--write-a1-cache']):
        with pytest.raises(ValueError,match='A5 must not receive'):
            validate_args(parser().parse_args(base+forbidden))


def test_parallel_a5_requires_checkpoint_provenance():
    args=parser().parse_args(['--mode','a5','--out','new-output','--native-manifest','train.json',
                              '--native-manifest-sha256','0'*64])
    with pytest.raises(ValueError,match='pinned frozen A0'):
        validate_args(args)
