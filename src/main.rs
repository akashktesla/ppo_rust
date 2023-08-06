#![allow(warnings)]
use std::default;
use rand::distributions::WeightedIndex;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tch::{nn,Tensor,Kind};
use tch::nn::{Adam,Module,VarStore,OptimizerConfig, Optimizer, Sequential};
use statrs::distribution::Categorical;

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
    vs:VarStore,
    actor:nn::Sequential,
    optimizer:Optimizer,
    checkpoint_file:String,
}

impl ActorNetwork{

    fn new(vs:VarStore,input_dims:i64,n_actions:i64)->ActorNetwork{
        let optimizer = Adam::default().build(&vs, 1e-3).unwrap();
        let path = vs.root();
        return ActorNetwork{
            actor:nn::seq()
                .add(nn::linear(vs.root(),input_dims,FC1_DIMS,Default::default()))
                .add_fn(|x|x.relu())
                .add(nn::linear(vs.root(),FC1_DIMS,FC2_DIMS,Default::default()))
                .add_fn(|x|x.relu())
                .add(nn::linear(vs.root(),FC2_DIMS,n_actions,Default::default()))
                .add_fn(|s|s.log_softmax(-1, Kind::Float)),
                optimizer,
                vs,
                checkpoint_file:"actor_torch_ppo".to_string(),
        };
    }

    fn forward(&self, xs: &Tensor) -> Categorical {
        let dist =  self.actor.forward(xs);
        let temp:Vec<f64> = dist.try_into().unwrap();
        let dist = Categorical::new(&temp[..]).unwrap();
        return dist;
    }

    fn save_checkpoint(&self){
        self.vs.save(self.checkpoint_file.clone());
    }
    fn load_checkpoint(&mut self){
        self.vs.load(self.checkpoint_file.clone());
    }

}

impl Module for ActorNetwork{
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.actor.forward(xs)
    }
}

struct CriticNetwork{
    vs:VarStore,
    checkpoint_file:String,
    critic:Sequential,
    optimizer:Optimizer,

}
impl CriticNetwork{
    fn new(vs:VarStore,input_dims:i64,alpha:f64)->CriticNetwork{
        return CriticNetwork { 
            checkpoint_file: String::from("critic_torch_ppo"),
            critic: nn::seq()
                .add(nn::linear(vs.root(), input_dims, FC1_DIMS, Default::default()))
                .add_fn(|x|x.relu())
                .add(nn::linear(vs.root(),FC1_DIMS,FC2_DIMS,Default::default()))
                .add_fn(|x|x.relu())
                .add(nn::linear(vs.root(),FC2_DIMS,1,Default::default())),
            optimizer : Adam::default().build(&vs, alpha).unwrap(),
            vs,
        };
    }

    fn forward(&self,x:Tensor)->Tensor{
        return self.critic.forward(&x);
    }

    fn save_checkpoint(&self){
        self.vs.save(self.checkpoint_file.clone());
    }
    
    fn load_checkpoint(&mut self){
        self.vs.load(self.checkpoint_file.clone());
    }

}

struct AgentBuilder{
    vs:VarStore,
    gamma:f64,
    alpha:f64,
    gae_lambda:f64,
    policy_clip:f64,
    batch_size:i64,
    n_epochs:i64,
    input_dims:i64,
    n_actions:i64,
}


struct Agent{
    gamma:f64,
    alpha:f64,
    gae_lambda:f64,
    policy_clip:f64,
    batch_size:i64,
    n_epochs:i64,
    actor:ActorNetwork,
    critic:CriticNetwork,
    memory:PPOMemory,
}



impl AgentBuilder{

    fn with_gamma(mut self,gamma:f64)->AgentBuilder{
        self.gamma = gamma;
        return self;
    }

    fn with_alpha(mut self,alpha:f64)->AgentBuilder{
        self.alpha = alpha;
        return self;
    }

    fn with_gae_lambda(mut self,gae_lambda:f64)->AgentBuilder{
        self.gae_lambda = gae_lambda;
        return self;
    }

    fn with_policy_clip(mut self,policy_clip:f64)->AgentBuilder{
        self.policy_clip = policy_clip;
        return self;
    }
    
    fn with_batch_size(mut self,batch_size:i64)->AgentBuilder{
        self.batch_size = batch_size;
        return self;
    }

    fn with_n_epochs(mut self,n_epochs:i64)->AgentBuilder{
        self.n_epochs = n_epochs;
        return self;
    }

    fn build(self)->Agent{
        let mut vs_2 = VarStore::new(tch::Device::Cpu);
        vs_2.copy(&self.vs);
        return Agent{
            gamma : self.gamma,
            alpha : self.alpha,
            gae_lambda : self.gae_lambda,
            policy_clip: self.policy_clip,
            batch_size: self.batch_size,
            n_epochs: self.n_epochs,
            actor: ActorNetwork::new(vs_2, self.input_dims, self.n_actions),
            critic: CriticNetwork::new(self.vs,self.input_dims,self.alpha),
            memory: PPOMemory::new(self.batch_size as i32),
        };
    }


}

impl Agent{
    fn new(vs:VarStore,n_actions:i64,input_dims:i64)->AgentBuilder{

        return AgentBuilder {  
            gamma : 0.99,
            alpha : 3e-3,
            gae_lambda : 0.95,
            policy_clip: 0.2,
            batch_size:64,
            n_epochs:10,
            input_dims,
             n_actions,
             vs,
        };
    }

    fn remember(&mut self,mut mem: PPOMemory){
        self.memory.store_memory(&mut mem);
    }

    fn save_models(&self){
        println!("... saving models ...");
        self.actor.save_checkpoint();
        self.critic.save_checkpoint();
    }

    fn load_models(&mut self){
        println!("... loading models ...");
        self.actor.load_checkpoint();
        self.critic.load_checkpoint();
    }

    fn choose_action(&self,observation:Vec<f64>){

    }
}






