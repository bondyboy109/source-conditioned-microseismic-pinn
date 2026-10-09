"""Train a constant-velocity time-shift MoE control on the development set."""
from __future__ import annotations
import hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from constant_velocity_moe import ConstantVelocityPrototypeMoE
from examples.run_traveltime_aligned_moe_development import (
    load_observations, sample_training_batch, aligned_prediction, relative_l2,
    base_prediction, source_loss_gradient,
)

PROTOCOL = ROOT / 'experiments/constant_velocity_moe_protocol.json'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    spec=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if len(sys.argv) > 2:
        spec['vavg_m_s'] = float(sys.argv[2])
    smoke=int(sys.argv[1]) if len(sys.argv)>1 else int(spec['training']['updates'])
    random.seed(spec['training']['seed']); np.random.seed(spec['training']['seed']); torch.manual_seed(spec['training']['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(spec['training']['seed'])
    manifest=torch.load(ROOT/'outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth', map_location='cpu', weights_only=False)
    paths=[Path(row['path']) for row in manifest['training_record']['receiver_observations']]
    train, receivers, times=load_observations(paths)
    dev_paths=sorted((ROOT/'data/observations/eikonal_independent').glob('*.npz'))+sorted((ROOT/'data/observations/eikonal_wide_search_independent').glob('*.npz'))
    dev, dev_receivers, dev_times=load_observations(dev_paths)
    if not np.array_equal(receivers,dev_receivers) or not np.array_equal(times,dev_times): raise ValueError('Acquisition mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domain=np.asarray([*spec['model']['spatial_lengths_m'],spec['model']['time_length_s']],dtype=np.float32)
    arrivals=np.zeros((len(train),len(receivers)),dtype=np.float32)
    model=ConstantVelocityPrototypeMoE(receivers,tuple(spec['model']['spatial_lengths_m']),spec['vavg_m_s'],time_length_s=spec['model']['time_length_s'],frequency_hz=spec['model']['frequency_hz'],expert_count=spec['model']['expert_count'],hidden_dim=spec['model']['hidden_dim'],hidden_layers=spec['model']['hidden_layers'],prototype_centers=torch.tensor(spec['model']['prototype_centers']),prototype_scales=tuple(spec['model']['prototype_scales']),temperature=spec['model']['prototype_temperature']).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=spec['training']['learning_rate']); rng=np.random.default_rng(spec['training']['seed']); history=[]; started=time.perf_counter()
    train_spec={'training':{'batch_size':spec['training']['batch_size'],'events_per_batch':spec['training']['events_per_batch'],'sampling':{'fmm_arrival_window_fraction':0.0,'observed_active_fraction':0.5,'uniform_fraction':0.5,'arrival_half_width_s':0.3}}}
    for update in range(1,smoke+1):
        points,sources,idx,target=sample_training_batch(train,receivers,times,arrivals,train_spec,rng,domain,device)
        opt.zero_grad(set_to_none=True); pred,gate,_=model(points,sources,idx); wave=(pred-target).square().mean(); balance=(gate.mean(0)-1/model.expert_count).square().mean(); sep=model.prototype_separation_loss(); loss=wave+0.01*balance+0.001*sep; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        with torch.no_grad(): model.prototype_centers.clamp_(0.05,0.95)
        if update==1 or update%100==0: history.append({'update':update,'loss':float(loss),'waveform':float(wave)}); print(history[-1],flush=True)
    tag=str(int(round(spec['vavg_m_s']))); out=ROOT/f'outputs/constant_velocity_moe_v{tag}'; out.mkdir(parents=True,exist_ok=True); ck=out/'constant_velocity_moe.pth'; torch.save({'state_dict':model.state_dict(),'protocol':spec,'training_seconds':time.perf_counter()-started,'training_updates':smoke},ck)
    print('saved',ck,flush=True)

if __name__=='__main__': main()
