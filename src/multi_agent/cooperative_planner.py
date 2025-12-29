"""
Cooperative Multi-Agent Planner with E-RECAP Integration.

This module implements the cooperative multi-agent planning setting where multiple
agents operate sequentially, each receiving a shared planning context that has been
pruned by E-RECAP's cost-aware token pruning module.
"""

import time
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .context_buffer import SharedPlanningContextBuffer, AgentContribution
from .structured_output import StructuredAgentOutput, build_structured_prompt
from .agent_config import AgentConfig, load_agent_configs
from .task_definitions import get_task_steps

# Import E-RECAP inference functions
# Note: This assumes cooperative_planner.py is in src/multi_agent/
# and inference_erecap.py is in src/
import sys
import os
# Add src/ to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from inference_erecap import load_model_and_pruners, prune_context_only


class CooperativeMultiAgentPlanner:
    """
    Cooperative multi-agent planner with E-RECAP token pruning.
    
    Implements the cooperative multi-agent setting where:
    - Multiple agents operate sequentially
    - Each agent receives a pruned shared planning context
    - Agent outputs are structured (observations, conflicts, plan patches)
    - Context accumulates over time and is pruned before each agent invocation
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        pruning_modules: torch.nn.ModuleDict,
        keep_ratio: float = 0.7,
        prune_layers: Optional[List[int]] = None,
        max_new_tokens: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the cooperative multi-agent planner.
        
        Args:
            model: Language model instance.
            tokenizer: Tokenizer instance.
            pruning_modules: Dictionary of pruning modules.
            keep_ratio: Fraction of tokens to keep per layer during pruning.
            prune_layers: List of layer indices to prune. If None, uses default.
            max_new_tokens: Maximum number of tokens to generate per agent.
            device: Device to run inference on. If None, uses model's device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pruning_modules = pruning_modules
        self.keep_ratio = keep_ratio
        self.prune_layers = prune_layers
        self.max_new_tokens = max_new_tokens
        self.device = device or next(model.parameters()).device
        
        self.context_buffer = SharedPlanningContextBuffer()
        self.agents: List[AgentConfig] = []
        self.planning_history: List[Dict] = []
    
    def add_agent(self, agent_config: AgentConfig):
        """Add an agent configuration to the planner."""
        self.agents.append(agent_config)
    
    def load_agents_from_config(self, config_path: Optional[str] = None):
        """Load agent configurations from file or use defaults."""
        self.agents = load_agent_configs(config_path)
    
    def _prune_context(self, context_text: str, use_pruning: bool = True) -> Tuple[str, Dict]:
        """
        Prune context using E-RECAP token pruning, or return original if baseline.
        
        Args:
            context_text: Full context text to be pruned.
            use_pruning: If True, apply E-RECAP pruning. If False, return original (baseline).
        
        Returns:
            pruned_text: Pruned context text (or original if baseline).
            pruning_stats: Pruning statistics (empty dict if baseline).
        """
        if not use_pruning:
            # Baseline: return original context without pruning
            return context_text, {"pruning_applied": False, "original_length": len(context_text)}
        
        pruned_text, pruning_stats = prune_context_only(
            model=self.model,
            tokenizer=self.tokenizer,
            pruning_modules=self.pruning_modules,
            input_text=context_text,
            keep_ratio=self.keep_ratio,
            prune_layers=self.prune_layers,
        )
        return pruned_text, pruning_stats
    
    def _call_agent_llm(
        self,
        prompt: str,
        agent_config: AgentConfig,
    ) -> str:
        """
        Call the language model for a single agent.
        
        Note: The context has already been pruned, so we use standard generation
        without additional pruning.
        
        Args:
            prompt: Input prompt for the agent.
            agent_config: Agent configuration.
        
        Returns:
            Generated text output from the agent.
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=10,  # Ensure minimum output length
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output (exclude input tokens)
        generated_ids = outputs[0][input_ids.shape[1]:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return output_text
    
    def _build_agent_prompt(
        self,
        agent_config: AgentConfig,
        pruned_context: str,
        task_step: Optional[str] = None,
    ) -> str:
        """
        Build the prompt for an agent based on pruned context.
        
        Args:
            agent_config: Agent configuration.
            pruned_context: Pruned context from previous agents.
            task_step: Optional specific task step description.
        
        Returns:
            Formatted prompt string.
        """
        return build_structured_prompt(
            agent_role=agent_config.role,
            agent_goal=agent_config.goal,
            agent_backstory=agent_config.backstory,
            pruned_context=pruned_context,
            task_step=task_step,
        )
    
    def run_planning_cycle(
        self,
        task_description: str,
        task_steps: Optional[List[Dict[str, str]]] = None,
        task_type: str = "cooperative",
        use_pruning: bool = True,
    ) -> Dict:
        """
        Execute a complete planning cycle with all agents.
        
        Args:
            task_description: Initial task description.
            task_steps: Optional list of task steps. If None, uses default.
            task_type: Type of task ("cooperative" or "embodied").
        
        Returns:
            Dictionary with planning results and statistics.
        """
        # Initialize context buffer
        self.context_buffer = SharedPlanningContextBuffer()
        self.context_buffer.set_task_description(task_description)
        self.planning_history = []
        
        # Load task steps if not provided
        if task_steps is None:
            task_steps = get_task_steps(task_type)
        
        # Ensure we have agents
        if not self.agents:
            self.load_agents_from_config()
        
        # If we don't have enough agents for all task steps, create agents dynamically from task steps
        if len(self.agents) < len(task_steps):
            # Create agent configs from task step roles
            for step_idx in range(len(self.agents), len(task_steps)):
                task_step = task_steps[step_idx]
                agent_role = task_step.get("agent_role", f"Agent {step_idx}")
                # Create a simple agent config for this role
                agent_config = AgentConfig(
                    agent_id=step_idx,
                    name=agent_role.replace(" ", ""),
                    role=agent_role,
                    goal=f"Complete the task step: {task_step.get('description', '')[:100]}",
                    backstory=f"You are a {agent_role} responsible for this planning step."
                )
                self.agents.append(agent_config)
        
        # Ensure number of agents matches task steps
        num_agents = min(len(self.agents), len(task_steps))
        
        start_time = time.time()
        total_pruning_time = 0.0
        total_inference_time = 0.0
        
        # Sequential agent execution
        for step_idx in range(num_agents):
            agent_config = self.agents[step_idx]
            task_step = task_steps[step_idx]
            
            step_start_time = time.time()
            
            # Get current context
            full_context = self.context_buffer.to_text()
            context_length_before = len(full_context)
            
            # Prune context using E-RECAP (or skip for baseline)
            prune_start = time.time()
            pruned_context, pruning_stats = self._prune_context(full_context, use_pruning=use_pruning)
            prune_time = time.time() - prune_start
            total_pruning_time += prune_time
            
            context_length_after = len(pruned_context)
            compression_ratio = context_length_after / context_length_before if context_length_before > 0 else 1.0
            
            # Build agent prompt
            agent_prompt = self._build_agent_prompt(
                agent_config=agent_config,
                pruned_context=pruned_context,
                task_step=task_step.get("description", None),
            )
            
            # Call agent LLM
            inference_start = time.time()
            llm_output = self._call_agent_llm(agent_prompt, agent_config)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Parse structured output
            structured_output = StructuredAgentOutput.from_llm_output(llm_output)
            
            # Create agent contribution
            contribution = AgentContribution(
                agent_id=agent_config.agent_id,
                agent_role=agent_config.role,
                observations=structured_output.observations,
                conflicts=structured_output.conflicts,
                plan_patches=structured_output.plan_patches,
            )
            
            # Update context buffer
            self.context_buffer.add_agent_contribution(contribution)
            
            step_time = time.time() - step_start_time
            
            # Record planning history
            self.planning_history.append({
                "step_id": step_idx,
                "agent_id": agent_config.agent_id,
                "agent_role": agent_config.role,
                "context_length_before": context_length_before,
                "context_length_after": context_length_after,
                "compression_ratio": compression_ratio,
                "pruning_time": prune_time,
                "inference_time": inference_time,
                "step_time": step_time,
                "pruning_stats": pruning_stats,
                "structured_output": structured_output.to_dict(),
            })
            
            print(f"[Step {step_idx}] {agent_config.role}: "
                  f"Context {context_length_before} -> {context_length_after} chars "
                  f"({compression_ratio:.2%}), "
                  f"Pruning {prune_time:.3f}s, Inference {inference_time:.3f}s")
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            "task_description": task_description,
            "num_agents": num_agents,
            "total_time": total_time,
            "total_pruning_time": total_pruning_time,
            "total_inference_time": total_inference_time,
            "planning_history": self.planning_history,
            "final_context_summary": self.context_buffer.get_summary(),
        }
        
        return results
    
    def get_final_context(self) -> str:
        """Get the final context buffer as text."""
        return self.context_buffer.to_text()
    
    def get_planning_summary(self) -> Dict:
        """Get a summary of the planning cycle."""
        return {
            "num_steps": len(self.planning_history),
            "total_time": sum(step["step_time"] for step in self.planning_history),
            "total_pruning_time": sum(step["pruning_time"] for step in self.planning_history),
            "total_inference_time": sum(step["inference_time"] for step in self.planning_history),
            "context_growth": [
                {
                    "step": step["step_id"],
                    "length_before": step["context_length_before"],
                    "length_after": step["context_length_after"],
                    "compression_ratio": step["compression_ratio"],
                }
                for step in self.planning_history
            ],
        }


def create_planner(
    model_path: str = "checkpoints/qwen2-7b-instruct",
    pruning_ckpt: str = "checkpoints/pruning_module.pt",
    keep_ratio: float = 0.7,
    prune_layers: Optional[List[int]] = None,
    max_new_tokens: int = 128,
) -> CooperativeMultiAgentPlanner:
    """
    Create a CooperativeMultiAgentPlanner instance with loaded model and pruners.
    
    Args:
        model_path: Path to the language model.
        pruning_ckpt: Path to the pruning module checkpoint.
        keep_ratio: Fraction of tokens to keep per layer.
        prune_layers: List of layer indices to prune. If None, uses default.
        max_new_tokens: Maximum tokens to generate per agent.
    
    Returns:
        Initialized CooperativeMultiAgentPlanner instance.
    """
    # Load model and pruning modules
    model, tokenizer, pruning_modules = load_model_and_pruners(prune_layers=prune_layers)
    
    # Create planner
    planner = CooperativeMultiAgentPlanner(
        model=model,
        tokenizer=tokenizer,
        pruning_modules=pruning_modules,
        keep_ratio=keep_ratio,
        prune_layers=prune_layers,
        max_new_tokens=max_new_tokens,
    )
    
    # Load default agents
    planner.load_agents_from_config()
    
    return planner

