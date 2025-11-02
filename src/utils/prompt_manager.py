"""
Centralized Prompt Template Management.

This module provides a PromptManager class that handles loading, caching,
and rendering of Jinja2 templates for all agent prompts.
"""

from pathlib import Path
from typing import Dict, Optional
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)


def find_prompts_dir() -> Path:
    """
    Find the prompts directory by traversing up from this file.
    
    Returns:
        Path to the prompts directory
        
    Raises:
        FileNotFoundError: If prompts directory cannot be found
    """
    current = Path(__file__).resolve()
    
    # Try parent directories up to 5 levels
    for _ in range(5):
        current = current.parent
        prompts_dir = current / "src" / "prompts"
        if prompts_dir.exists() and prompts_dir.is_dir():
            return prompts_dir
    
    # Fallback: try relative to package root
    package_root = Path(__file__).resolve().parent.parent
    prompts_dir = package_root / "prompts"
    if prompts_dir.exists():
        return prompts_dir
    
    raise FileNotFoundError("Could not locate prompts directory")


class PromptManager:
    """
    Manages loading and rendering of prompt templates.
    
    Features:
    - Template caching for performance
    - Automatic template discovery
    - Consistent error handling
    - Template validation
    
    Example:
        >>> pm = PromptManager()
        >>> system_prompt = pm.render("plan_system", has_system_elements=True, K=3)
        >>> user_prompt = pm.render("plan_user", goal="Build an app", tasks=[...])
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize the PromptManager.
        
        Args:
            prompts_dir: Optional path to prompts directory. If None, auto-discovers.
        """
        self.prompts_dir = prompts_dir or find_prompts_dir()
        self._cache: Dict[str, Template] = {}
        logger.info(f"PromptManager initialized with directory: {self.prompts_dir}")
    
    def get_template(self, name: str) -> Template:
        """
        Get a template by name, loading and caching it if necessary.
        
        Args:
            name: Template name without extension (e.g., "plan_system")
            
        Returns:
            Jinja2 Template object
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            Exception: If template parsing fails
        """
        if name not in self._cache:
            template_path = self.prompts_dir / f"{name}.jinja"
            
            if not template_path.exists():
                raise FileNotFoundError(
                    f"Template '{name}' not found at {template_path}"
                )
            
            try:
                template_content = template_path.read_text(encoding='utf-8')
                self._cache[name] = Template(template_content)
                logger.debug(f"Loaded and cached template: {name}")
            except Exception as e:
                logger.error(f"Failed to load template '{name}': {e}")
                raise
        
        return self._cache[name]
    
    def render(self, name: str, **kwargs) -> str:
        """
        Render a template with the given variables.
        
        Args:
            name: Template name without extension
            **kwargs: Variables to pass to the template
            
        Returns:
            Rendered template string
            
        Raises:
            FileNotFoundError: If template doesn't exist
            Exception: If rendering fails
        """
        try:
            template = self.get_template(name)
            rendered = template.render(**kwargs)
            logger.debug(f"Rendered template '{name}' with {len(kwargs)} variables")
            return rendered
        except Exception as e:
            logger.error(f"Failed to render template '{name}': {e}")
            raise
    
    def clear_cache(self):
        """Clear the template cache."""
        self._cache.clear()
        logger.info("Template cache cleared")
    
    def list_templates(self) -> list[str]:
        """
        List all available template names.
        
        Returns:
            List of template names (without .jinja extension)
        """
        templates = []
        for template_file in self.prompts_dir.glob("*.jinja"):
            templates.append(template_file.stem)
        return sorted(templates)
    
    def template_exists(self, name: str) -> bool:
        """
        Check if a template exists.
        
        Args:
            name: Template name without extension
            
        Returns:
            True if template exists, False otherwise
        """
        template_path = self.prompts_dir / f"{name}.jinja"
        return template_path.exists()
    
    def reload_template(self, name: str):
        """
        Force reload a template from disk, bypassing cache.
        
        Args:
            name: Template name without extension
        """
        if name in self._cache:
            del self._cache[name]
        self.get_template(name)  # Reload from disk
        logger.info(f"Reloaded template: {name}")


# Global singleton instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """
    Get the global PromptManager singleton instance.
    
    Returns:
        Global PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def render_prompt(name: str, **kwargs) -> str:
    """
    Convenience function to render a prompt using the global manager.
    
    Args:
        name: Template name without extension
        **kwargs: Variables to pass to the template
        
    Returns:
        Rendered template string
    """
    return get_prompt_manager().render(name, **kwargs)
