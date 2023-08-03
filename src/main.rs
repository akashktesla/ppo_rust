#![allow(warnings)]
use std::default;
use rand::distributions::WeightedIndex;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tch::{nn,Tensor,Kind};
use tch::nn::{Adam,Module,VarStore,OptimizerConfig, Optimizer};

const FC1_DIMS:i64 = 256;
const FC2_DIMS:i64 = 256;

fn main() {
    let vs = nn::VarStore::new(tch::Device::Cpu);
}

struct PPOMemory{
    states:Vec<i32>,
    probs:Vec<i32>,
    vals:Vec<i32>,
    actions:Vec<i32>,
    rewards:Vec<i32>,
    dones:Vec<i32>,
    batch_size:i32,
}

impl PPOMemory{
    fn new(batch_size:i32)->PPOMemory{
        return PPOMemory { 
            states: Vec::new(),
            probs: Vec::new(),
            vals: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            dones: Vec::new(),
            batch_size 
        }
    }
    fn generate_batches(&self)->(&PPOMemory,Vec<Vec<i64>>){
        let n_states = self.states.len();
        let mut batch_start:Vec<i32> = (0..n_states as i32).step_by(self.batch_size as usize).collect();
        let indices:Vec<i64> = (0..n_states as i64).collect();
        let mut rng = thread_rng();
        batch_start.shuffle(&mut rng);
        let batches:Vec<Vec<i64>> = batch_start.iter()
            .map(|i|{
                return indices[*i as usize ..(*i+self.batch_size) as usize].to_vec();
            })
            .collect();
            return (self,batches);
    }
    fn store_memory(&mut self,ppomem:&mut PPOMemory){
        self.states.append(&mut ppomem.states);
        self.probs.append(&mut ppomem.probs);
        self.vals.append(&mut ppomem.vals);
        self.rewards.append(&mut ppomem.rewards);
        self.dones.append(&mut ppomem.dones);
    }
    fn clear_memory(&mut self){
        self.states = Vec::new();
        self.probs = Vec::new();
        self.actions = Vec::new();
        self.rewards = Vec::new();
        self.dones = Vec::new();
        self.vals = Vec::new();
    }
}

#[derive(Debug)]
struct ActorNetwork{
    actor:nn::Sequential,
    optimizer:Optimizer,
}

impl ActorNetwork{
    fn new(vs:&VarStore,input_dims:i64,n_actions:i64)->ActorNetwork{
        let optimizer = Adam::default().build(vs, 1e-3).unwrap();
        return ActorNetwork{
            actor:nn::seq()
                .add(nn::linear(vs.root(),input_dims,FC1_DIMS,Default::default()))
                .add_fn(|x|x.relu())
                .add(nn::linear(vs.root(),FC1_DIMS,FC2_DIMS,Default::default()))
                .add_fn(|x|x.relu())
                .add(nn::linear(vs.root(),FC2_DIMS,n_actions,Default::default()))
                .add_fn(|s|s.log_softmax(-1, Kind::Float)),
                optimizer
        };
    }
}

impl Module for ActorNetwork{
    fn forward(&self, xs: &Tensor) -> Tensor {
       let dist =  self.actor.forward(xs);
       return dist;
    }
}
