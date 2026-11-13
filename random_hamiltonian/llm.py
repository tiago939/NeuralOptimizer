"""ITQPP strict common prompt: QASM + Pauli Hamiltonian + optional noise + energy history.

No problem name, instance metadata, hidden angles, reference energy, initial
angles, optimization hints, or classical baseline outcomes are sent to the LLM.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
import numpy as np
from circuit import canonical_angles, export_qasm3, parameter_names
from hamiltonian import make_problem, EnergyOracle, descriptions
from noise import noise_spec

from prompt import SYSTEM, build_user_prompt


def parse_proposal(raw,gates):
    try: obj=json.loads(raw.strip())
    except json.JSONDecodeError:
        match=re.search(r'\{.*\}',raw,re.S)
        if not match:raise ValueError(f'Missing JSON response: {raw[:500]}')
        obj=json.loads(match.group())
    return canonical_angles(obj['theta'],gates),str(obj.get('mechanism',''))


def log(message):print(f'[{time.strftime("%H:%M:%S")}] {message}',file=sys.stderr,flush=True)


def save(path,data):
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,indent=2),encoding='utf8')
    log(f'Saved {dest}')


def run(n=8,layers=2,active_per_layer=2,k=128,seed=42,max_depth=100,
        max_attempts=250,rounds=1,shots=0,noise=False,noise_prompt='blind',
        model=None,shot_seed=2026,output=None,heartbeat_seconds=10,api_timeout=300,
        non_clifford_count=1):
    from openai import OpenAI
    if noise_prompt not in ('blind','aware'):raise ValueError('noise_prompt must be blind or aware')
    if noise_prompt=='aware' and not noise:raise ValueError('--noise-prompt aware requires --noise')
    if rounds<1:raise ValueError('rounds must be >=1')
    model=model or os.getenv('OPENAI_MODEL')
    if not model or not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY and --model or OPENAI_MODEL')
    problem=make_problem(n,layers,active_per_layer,k,seed,max_depth,max_attempts,
                         non_clifford_count=non_clifford_count)
    oracle=EnergyOracle(problem,shots,noise,shot_seed)
    qasm=export_qasm3(problem.gates,n)
    # No initial-energy query: the first LLM round is truly zero-feedback.
    history=[];trace=[];prompts=[]
    file=output or f'results/llm_{problem.instance_id}_{"noisy" if noise else "ideal"}_{noise_prompt}_k{shots}.json'
    client=OpenAI(timeout=api_timeout,max_retries=0)
    log(f'Parent benchmark {problem.instance_id} | n={n}, M={len(problem.terms)}, depth={problem.native_depth}, parameters={len(parameter_names(problem.gates))}')
    for round_id in range(1,rounds+1):
        prompt=build_user_prompt(qasm,descriptions(problem.terms),history,
                 noise_spec() if noise and noise_prompt=='aware' else None)
        prompts.append(prompt)
        log(f'Round {round_id}/{rounds}: sending {len(prompt)} characters to {model}')
        done=threading.Event()
        def heartbeat():
            start=time.monotonic()
            while not done.wait(heartbeat_seconds):
                log(f'Model API pending: {time.monotonic()-start:.0f} seconds')
        thread=None
        if heartbeat_seconds>0:
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        start=time.monotonic()
        try:response=client.responses.create(model=model,instructions=SYSTEM,input=prompt)
        finally:
            done.set()
            if thread:thread.join(timeout=1)
        theta,mechanism=parse_proposal(response.output_text,problem.gates)
        measured=oracle.energy(theta)
        diagnostic=oracle.ideal_unbilled(theta) # NEVER sent to model.
        history.append({'theta':theta.tolist(),'energy':float(measured)})
        usage=getattr(response,'usage',None)
        trace.append({'step':round_id,'evaluations':oracle.evaluations,'shots':oracle.total_shots,
                      'estimated_energy':measured,'exact_energy':diagnostic,
                      'exact_gap':max(0.,diagnostic),'theta':theta.tolist(),
                      'mechanism':mechanism,'response_id':getattr(response,'id',None),
                      'input_tokens':getattr(usage,'input_tokens',None),
                      'output_tokens':getattr(usage,'output_tokens',None),
                      'inference_seconds':time.monotonic()-start})
        data={'method':'llm_direct_absolute','problem':'parent_hamiltonian','model':model,
              'instance_id':problem.instance_id,'config':{'n':n,'layers':layers,'active_per_layer':active_per_layer,
              'k':k,'seed':seed,'max_depth':max_depth,'non_clifford_count':non_clifford_count,
               'shots_per_energy_query':shots,
              'noise':noise,'noise_prompt':noise_prompt,'shot_seed':shot_seed},
              'hamiltonian':descriptions(problem.terms),'native_depth':problem.native_depth,
              'ground_energy':0.0,'trace':trace,'system_prompt':SYSTEM,'user_prompts':prompts,
              # Private benchmark metadata is stored locally, NEVER in the API prompt.
              'private':{'hidden_theta':problem.hidden_theta.tolist(),'topology_seed':problem.topology_seed,
                         'generation_attempts':problem.attempts,
                          'non_clifford_indices':list(problem.non_clifford_indices)}}
        save(file,data)
        log(f'E_observed={measured:.9g}, ideal E={diagnostic:.9g}, queries={oracle.evaluations}, shots={oracle.total_shots}')
    return data


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=4)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--active-per-layer',type=int,default=None)
    p.add_argument('--k',type=int,default=6)
    p.add_argument('--non-clifford-count',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-depth',type=int,default=100)
    p.add_argument('--max-attempts',type=int,default=250)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--shots',type=int,default=200)
    p.add_argument('--noise',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--noise-prompt',choices=['blind','aware'],default='blind')
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--shot-seed',type=int,default=2026)
    p.add_argument('--output',default=None)
    p.add_argument('--heartbeat-seconds',type=float,default=10)
    p.add_argument('--api-timeout',type=float,default=300)
    run(**vars(p.parse_args()))

if __name__=='__main__':cli()
