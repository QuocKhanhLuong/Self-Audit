#!/usr/bin/env python3
"""Wake this existing Codex thread once final experiment evidence is ready.

No GitHub credentials are persisted. Only the existing local app-server Unix
socket is used. The resumed thread retains its authorized GitHub connector and
must verify evidence before publishing; this watcher never claims a push itself.
Protocol: https://learn.chatgpt.com/docs/app-server
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time


class LocalRpc:
    def __init__(self, path):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.sock.settimeout(15);self.sock.connect(str(path));self.counter=0
        key=base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(('GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n'
            'Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: '+key+'\r\n\r\n').encode())
        header=bytearray()
        while not header.endswith(b'\r\n\r\n'):
            header.extend(self.exact(1))
            if len(header)>16384:raise RuntimeError('oversized local socket handshake')
        expected=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
        lines=header.decode().split('\r\n')
        headers={k.strip().lower():v.strip() for line in lines[1:] if ':' in line for k,v in [line.split(':',1)]}
        if '101' not in lines[0] or headers.get('sec-websocket-accept')!=expected:
            raise RuntimeError('local socket did not accept WebSocket upgrade')
        self.rpc('initialize',dict(clientInfo=dict(name='self_audit_completion_watcher',version='1.0'),
                                  capabilities=dict(experimentalApi=True)))
        self.send(dict(method='initialized'))

    def exact(self,n):
        chunks=bytearray()
        while len(chunks)<n:
            data=self.sock.recv(n-len(chunks))
            if not data:raise ConnectionError('local socket closed')
            chunks.extend(data)
        return bytes(chunks)

    def frame(self,opcode,payload):
        size=len(payload);mask=os.urandom(4)
        if size<126:head=bytes((0x80|opcode,0x80|size))
        elif size<65536:head=bytes((0x80|opcode,0x80|126))+struct.pack('!H',size)
        else:head=bytes((0x80|opcode,0x80|127))+struct.pack('!Q',size)
        self.sock.sendall(head+mask+bytes(b^mask[i%4] for i,b in enumerate(payload)))

    def send(self,obj):self.frame(1,json.dumps(obj).encode())

    def receive(self):
        message=bytearray()
        while True:
            first,second=self.exact(2);opcode=first&15;size=second&127
            if size==126:size=struct.unpack('!H',self.exact(2))[0]
            elif size==127:size=struct.unpack('!Q',self.exact(8))[0]
            if size>32*2**20:raise RuntimeError('oversized local RPC response')
            mask=self.exact(4) if second&128 else None
            payload=self.exact(size)
            if mask:payload=bytes(b^mask[i%4] for i,b in enumerate(payload))
            if opcode==8:raise ConnectionError('local WebSocket closed')
            if opcode==9:self.frame(10,payload);continue
            if opcode==10:continue
            if opcode not in (0,1):raise RuntimeError('unsupported local WebSocket frame')
            message.extend(payload)
            if first&128:return json.loads(message)

    def rpc(self,method,params):
        self.counter+=1;identifier=self.counter
        self.send(dict(id=identifier,method=method,params=params))
        while True:
            message=self.receive()
            if message.get('id')!=identifier:continue
            if 'error' in message:raise RuntimeError(message['error'])
            return message['result']

    def close(self):self.sock.close()


def completion_event(root):
    state=json.loads((root/'STATUS.json').read_text())
    if state['stage']=='FAILED':return 'FAILED_REQUIRES_INSPECTION'
    if state['stage']!='COMPLETED':return None
    rows=json.loads((root/'RESULTS.json').read_text())
    required={(f'A{i}',split,checkpoint) for i in range(7) for split,checkpoint in
              (('VAL18','step_005000'),('VAL18','final'),('TEST50','final'))}
    actual={(r['arm'],r['split'],r['checkpoint']) for r in rows}
    if not required<=actual:raise RuntimeError('COMPLETED without all 21 expected frozen evaluations')
    for name in ('check-repo-final.log','pytest-final.log','TRAINING_RESULTS.json',
                 'FROZEN_FINAL_CHECKPOINTS.json','RESULTS.csv'):
        if not (root/name).is_file():raise RuntimeError('completion evidence missing: '+name)
    return 'COMPLETED_VERIFY_AND_PUBLISH'


def run(root,thread_id,socket_path,probe=False):
    def status(stage,**kwargs):
        obj=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                 thread_id=thread_id,**kwargs)
        if not probe:
            target=root/'AUTO_PUBLISH.json';tmp=target.with_suffix('.tmp')
            tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,target)
        print(json.dumps(obj),flush=True)
    rpc=LocalRpc(socket_path)
    try:
        thread=rpc.rpc('thread/read',dict(threadId=thread_id,includeTurns=False))['thread']
        assert thread['id']==thread_id
        status('CONNECTION_VERIFIED',thread_status=thread.get('status'))
    finally:rpc.close()
    if probe:return
    status('WAITING_FOR_EXPERIMENT_COMPLETION',artifact_root=str(root),automatic_push_authorized=True)
    while True:
        try:
            if (root/'GIT_PUBLICATION.json').exists():
                result=json.loads((root/'GIT_PUBLICATION.json').read_text())
                if result.get('status')=='PUSHED':status('PUSHED',publication=result);return
            event=completion_event(root)
            if event is None:time.sleep(30);continue
            rpc=LocalRpc(socket_path)
            try:
                thread=rpc.rpc('thread/read',dict(threadId=thread_id,includeTurns=False))['thread']
                live=thread.get('status',{})
                if isinstance(live,dict) and live.get('type')=='active':time.sleep(10);continue
                # Read-only metadata above never changes the session. Resume only
                # when terminal evidence is ready, and inherit every model/policy.
                rpc.rpc('thread/resume',dict(threadId=thread_id,excludeTurns=True))
                text=(f'Automatic completion watcher for the user-authorized Self-Audit run: {event}. '
                      f'Artifacts: {root}. Continue the existing task; the user explicitly requested '
                      f'Code worktree: {Path(__file__).resolve().parents[1]}. '
                      'automatic Git push when finished. Inspect STATUS.json and final evidence. '
                      'If complete, publish all seven training/VAL/TEST results, negative findings, '
                      'benchmark tables, configs, source/weight/checkpoint hashes and compact logs '
                      'to experiments/teacher-memory-full-20261024 using the authenticated GitHub connector. '
                      'Do not add patient arrays, predictions or checkpoint weights to Git. Preserve '
                      'uncommitted user changes and never force-push. Run required checks before delivery. '
                      'After verifying the remote commit, write GIT_PUBLICATION.json in the artifact root '
                      'with status PUSHED, branch and commit SHA; report Dice including BG and patient macro. '
                      'If failed, report the actual failure and repair only within existing authorization; '
                      'do not publish an incomplete run as completed.')
                result=rpc.rpc('turn/start',dict(threadId=thread_id,input=[dict(type='text',text=text)]))
                turn=result['turn']['id']
                status('PUBLICATION_TURN_STARTED',turn_id=turn,event=event)
                # Keep transport alive until the accepted turn terminates so its
                # lifecycle and connector tool responses stay attached normally.
                while True:
                    try:message=rpc.receive()
                    except socket.timeout:continue
                    if message.get('method')=='turn/completed' and message.get('params',{}).get('turn',{}).get('id')==turn:
                        result=message['params']['turn']
                        receipt=root/'GIT_PUBLICATION.json'
                        publication=json.loads(receipt.read_text()) if receipt.exists() else None
                        stage='PUSHED' if publication and publication.get('status')=='PUSHED' else 'PUBLICATION_TURN_FINISHED_WITHOUT_PUSH_RECEIPT'
                        status(stage,turn_id=turn,turn_status=result['status'],publication=publication)
                        return
            finally:rpc.close()
        except Exception as exc:
            status('RETRYING',error=str(exc));time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--thread-id',required=True);p.add_argument('--socket',type=Path,
        default=Path('/root/.codex/app-server-control/app-server-control.sock'))
    p.add_argument('--probe',action='store_true');a=p.parse_args()
    run(a.root.resolve(),a.thread_id,a.socket,a.probe)
