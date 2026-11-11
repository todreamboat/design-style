"""Wall-clock bounded calls; persistent reservations survive terminated workers."""
import multiprocessing as mp
import json
from run_p1_ablation import PersistentBudget
import workflow_benchmark as w


def _request_worker(conn, out, cap, args):
    try:
        budget=PersistentBudget(out,cap)
        result=budget.call(*args)
        conn.send(('ok',result))
    except Exception as exc:
        conn.send(('error',type(exc).__name__))
    finally:
        conn.close()


def bounded_worker(target,args,seconds):
    if seconds<=0:
        raise ValueError('Positive deadline required')
    ctx=mp.get_context('spawn')
    parent,child=ctx.Pipe(duplex=False)
    proc=ctx.Process(target=target,args=(child,*args),daemon=True)
    proc.start()
    child.close()
    try:
        if not parent.poll(seconds):
            raise TimeoutError('API wall-clock deadline reached; no automatic retry')
        try:
            state,value=parent.recv()
        except EOFError:
            raise RuntimeError('API worker exited without result') from None
        if state!='ok':
            raise RuntimeError('API worker stopped: '+value)
        return value
    finally:
        parent.close()
        proc.join(.2)
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        proc.close()


class DeadlineBudget(PersistentBudget):
    def call(self,kind,cfg,prompt,paths,seed=None):
        try:
            return bounded_worker(_request_worker,(self.out,self.cap,(kind,cfg,prompt,paths,seed)),300)
        finally:
            # Failed calls without usage retain the original conservative reservation.
            ledger=self.out/'budget_ledger.json'
            if ledger.exists():
                saved=w.load(ledger)
                if saved['cap_usd']!=self.cap:
                    raise ValueError('Budget changed during request')
                self.rows=saved['requests']


class AuthorizedRetryBudget(DeadlineBudget):
    """At most one unchanged network retry, persisted across continuations."""
    def fingerprint(self,kind,cfg,prompt,paths,seed=None):
        return w.text_hash(json.dumps([kind,cfg['image_model'] if kind=='image' else cfg['vlm_model'],
                                      prompt,[str(p) for p in paths],seed],ensure_ascii=False))

    def retry_state(self):
        path=self.out/'network_retry_state.json'
        return w.load(path) if path.exists() else {}

    def mark_next_as_retry(self,kind,cfg,prompt,paths,seed=None):
        states=self.retry_state()
        key=self.fingerprint(kind,cfg,prompt,paths,seed)
        if key not in states:
            states[key]=dict(retry_used=True,exhausted=False,reason='resume of previously failed network dispatch')
            w.save(self.out/'network_retry_state.json',states)

    def _network_failure(self,exc,before):
        if len(self.rows)<=before:
            return False
        if isinstance(exc,TimeoutError):
            return True
        directory=self.out/'calls'/f'request_{len(self.rows):03d}'
        for path in directory.glob('*.error.json'):
            error=w.load(path)
            if error.get('error')=='network_error' and 'PermissionError' not in error.get('exception_chain_types',[]):
                return True
        return False

    def call(self,kind,cfg,prompt,paths,seed=None):
        key=self.fingerprint(kind,cfg,prompt,paths,seed)
        states=self.retry_state()
        state=states.get(key,dict(retry_used=False,exhausted=False))
        if state['exhausted']:
            raise RuntimeError('Authorized network retry already exhausted; no dispatch')
        while True:
            before=len(self.rows)
            try:
                return super().call(kind,cfg,prompt,paths,seed)
            except (RuntimeError,TimeoutError) as exc:
                if not self._network_failure(exc,before):
                    raise
                if state['retry_used']:
                    state['exhausted']=True
                    states[key]=state
                    w.save(self.out/'network_retry_state.json',states)
                    raise
                state['retry_used']=True
                states[key]=state
                w.save(self.out/'network_retry_state.json',states)
                print('Network failure: one user-authorized unchanged retry; reservation retained',flush=True)
