"""
State validation using Pydantic models.

This module provides validation for state transitions to ensure data integrity
throughout the agent workflow.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)


class SubTaskModel(BaseModel):
    """Validation model for subtasks."""
    id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., min_length=1, description="Task description")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Task dependencies")
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Task ID cannot be empty")
        return v.strip()
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError("Task description must be at least 3 characters")
        return v.strip()


class PluginModel(BaseModel):
    """Validation model for plugins."""
    name: str = Field(..., min_length=1, description="Plugin name")
    goal: Optional[str] = Field(None, description="Plugin goal/purpose")
    description: Optional[str] = Field(None, description="Plugin description")
    tools: Optional[List[str]] = Field(default_factory=list, description="Available tools")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="Plugin dependencies")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Plugin name cannot be empty")
        return v.strip()


class PlanNodeModel(BaseModel):
    """Validation model for plan nodes."""
    id: str = Field(..., description="Node identifier")
    prompt: Optional[str] = Field(None, description="Node prompt/instruction")
    tool: Optional[str] = Field(None, description="Tool to execute")
    agent: Optional[str] = Field(None, description="Agent to use")
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Node ID cannot be empty")
        return v.strip()


class PlanEdgeModel(BaseModel):
    """Validation model for plan edges."""
    from_node: Union[str, int] = Field(..., alias="from", description="Source node")
    to: Union[str, int] = Field(..., description="Target node")
    condition: Optional[str] = Field(None, description="Edge condition")
    
    @field_validator('from_node', 'to')
    @classmethod
    def validate_node_ref(cls, v: Union[str, int]) -> Union[str, int]:
        if isinstance(v, str) and not v.strip():
            raise ValueError("Node reference cannot be empty string")
        return v


class PlanModel(BaseModel):
    """Validation model for execution plans."""
    nodes: List[PlanNodeModel] = Field(..., min_length=1, description="Plan nodes")
    edges: Optional[List[PlanEdgeModel]] = Field(default_factory=list, description="Plan edges")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Plan confidence score")
    reasoning: Optional[str] = Field(None, description="Planning reasoning")
    
    @field_validator('nodes')
    @classmethod
    def validate_nodes(cls, v: List[PlanNodeModel]) -> List[PlanNodeModel]:
        if not v:
            raise ValueError("Plan must have at least one node")
        
        # Check for duplicate node IDs
        node_ids = [node.id for node in v]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node IDs found in plan")
        
        return v
    
    @model_validator(mode='after')
    def validate_edges(self) -> 'PlanModel':
        """Validate that all edge references point to existing nodes."""
        if not self.edges:
            return self
        
        node_ids = {node.id for node in self.nodes}
        
        for edge in self.edges:
            if str(edge.from_node) not in node_ids:
                raise ValueError(f"Edge references non-existent source node: {edge.from_node}")
            if str(edge.to) not in node_ids:
                raise ValueError(f"Edge references non-existent target node: {edge.to}")
        
        return self


class ReviewModel(BaseModel):
    """Validation model for HITL review."""
    action: str = Field(..., pattern="^(approve|revise|reject)$", description="Review action")
    feedback: Optional[str] = Field(None, description="Review feedback")
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid_actions = {"approve", "revise", "reject"}
        if v not in valid_actions:
            raise ValueError(f"Action must be one of {valid_actions}")
        return v


class CostEstimateModel(BaseModel):
    """Validation model for cost estimates."""
    llm_calls: float = Field(0.0, ge=0.0, description="Estimated LLM cost")
    tool_execution: float = Field(0.0, ge=0.0, description="Estimated tool cost")
    estimated_time: float = Field(0.0, ge=0.0, description="Estimated time in seconds")
    total_cost: float = Field(0.0, ge=0.0, description="Total estimated cost")


class StateValidator:
    """
    Validates state transitions and data integrity.
    
    Example:
        >>> validator = StateValidator()
        >>> validator.validate_split_tasks(state)
        >>> validator.validate_plan(state)
    """
    
    @staticmethod
    def validate_split_tasks(state: Dict[str, Any]) -> List[str]:
        """
        Validate split tasks in state.
        
        Args:
            state: Application state
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        split_tasks = state.get("split_tasks", [])
        
        if not isinstance(split_tasks, list):
            errors.append("split_tasks must be a list")
            return errors
        
        for i, task in enumerate(split_tasks):
            try:
                SubTaskModel(**task)
            except Exception as e:
                errors.append(f"Task {i}: {str(e)}")
        
        return errors
    
    @staticmethod
    def validate_plugins(state: Dict[str, Any]) -> List[str]:
        """
        Validate plugins in state.
        
        Args:
            state: Application state
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        plugins = state.get("plugins", [])
        
        if not isinstance(plugins, list):
            errors.append("plugins must be a list")
            return errors
        
        for i, plugin in enumerate(plugins):
            try:
                PluginModel(**plugin)
            except Exception as e:
                errors.append(f"Plugin {i}: {str(e)}")
        
        return errors
    
    @staticmethod
    def validate_plan(state: Dict[str, Any]) -> List[str]:
        """
        Validate execution plan in state.
        
        Args:
            state: Application state
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        plan = state.get("plan")
        
        if plan is None:
            errors.append("plan is None")
            return errors
        
        if not isinstance(plan, dict):
            errors.append("plan must be a dictionary")
            return errors
        
        try:
            PlanModel(**plan)
        except Exception as e:
            errors.append(f"Plan validation error: {str(e)}")
        
        return errors
    
    @staticmethod
    def validate_review(state: Dict[str, Any]) -> List[str]:
        """
        Validate HITL review in state.
        
        Args:
            state: Application state
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        review = state.get("review")
        
        if review is None:
            return errors  # Review is optional
        
        if not isinstance(review, dict):
            errors.append("review must be a dictionary")
            return errors
        
        try:
            ReviewModel(**review)
        except Exception as e:
            errors.append(f"Review validation error: {str(e)}")
        
        return errors
    
    @staticmethod
    def validate_cost_estimate(state: Dict[str, Any]) -> List[str]:
        """
        Validate cost estimate in state.
        
        Args:
            state: Application state
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        cost_estimate = state.get("cost_estimate")
        
        if cost_estimate is None:
            return errors  # Cost estimate is optional
        
        if not isinstance(cost_estimate, dict):
            errors.append("cost_estimate must be a dictionary")
            return errors
        
        try:
            CostEstimateModel(**cost_estimate)
        except Exception as e:
            errors.append(f"Cost estimate validation error: {str(e)}")
        
        return errors
    
    @staticmethod
    def validate_state(state: Dict[str, Any], strict: bool = False) -> Dict[str, List[str]]:
        """
        Validate entire state.
        
        Args:
            state: Application state
            strict: If True, raises exception on validation errors
            
        Returns:
            Dictionary mapping field names to validation errors
            
        Raises:
            ValueError: If strict=True and validation fails
        """
        all_errors = {}
        
        # Validate each component
        if "split_tasks" in state:
            errors = StateValidator.validate_split_tasks(state)
            if errors:
                all_errors["split_tasks"] = errors
        
        if "plugins" in state:
            errors = StateValidator.validate_plugins(state)
            if errors:
                all_errors["plugins"] = errors
        
        if "plan" in state:
            errors = StateValidator.validate_plan(state)
            if errors:
                all_errors["plan"] = errors
        
        if "review" in state:
            errors = StateValidator.validate_review(state)
            if errors:
                all_errors["review"] = errors
        
        if "cost_estimate" in state:
            errors = StateValidator.validate_cost_estimate(state)
            if errors:
                all_errors["cost_estimate"] = errors
        
        if strict and all_errors:
            error_msg = "\n".join([
                f"{field}: {', '.join(errors)}"
                for field, errors in all_errors.items()
            ])
            raise ValueError(f"State validation failed:\n{error_msg}")
        
        return all_errors
    
    @staticmethod
    def check_circular_dependencies(tasks: List[Dict[str, Any]]) -> Optional[List[str]]:
        """
        Check for circular dependencies in tasks.
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            List of task IDs in cycle, or None if no cycle exists
        """
        # Build adjacency list
        graph = {}
        for task in tasks:
            task_id = task.get("id")
            deps = task.get("dependencies", [])
            graph[task_id] = deps
        
        # DFS to detect cycle
        visited = set()
        rec_stack = set()
        
        def has_cycle(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                result = has_cycle(node, [])
                if result:
                    return result
        
        return None


def validate_state_transition(state: Dict[str, Any], log_errors: bool = True) -> bool:
    """
    Convenience function to validate state.
    
    Args:
        state: Application state
        log_errors: Whether to log validation errors
        
    Returns:
        True if state is valid, False otherwise
    """
    validator = StateValidator()
    errors = validator.validate_state(state, strict=False)
    
    if errors and log_errors:
        for field, field_errors in errors.items():
            for error in field_errors:
                logger.warning(f"State validation error in {field}: {error}")
    
    return len(errors) == 0
