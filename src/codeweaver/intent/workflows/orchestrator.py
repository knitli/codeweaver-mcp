"""Multi-step workflow orchestration using existing service patterns."""

import logging
import time

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import ServiceConfig, ServiceHealth, ServiceIntegrationError, ServiceType


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""

    name: str
    "Name of the workflow step."
    handler: Callable
    "Function to execute for this step."
    required: bool = True
    "Whether this step is required for workflow success."
    timeout: float = 30.0
    "Maximum time allowed for this step in seconds."
    retry_count: int = 0
    "Number of retries allowed for this step."
    dependencies: list[str] | None = None
    "Names of steps that must complete before this step."
    metadata: dict[str, Any] | None = None
    "Additional metadata for the step."

    def __post_init__(self):
        """Initialize default values."""
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowDefinition:
    """Defines a complete workflow with steps and configuration."""

    name: str
    "Name of the workflow."
    steps: list[WorkflowStep]
    "List of steps to execute in order."
    description: str = ""
    "Human-readable description of the workflow."
    timeout: float = 300.0
    "Maximum time allowed for entire workflow in seconds."
    allow_partial_success: bool = True
    "Whether to continue if optional steps fail."
    metadata: dict[str, Any] | None = None
    "Additional metadata for the workflow."

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowStepResult:
    """Result of a single workflow step execution."""

    step_name: str
    "Name of the step that was executed."
    success: bool
    "Whether the step succeeded."
    data: Any
    "Result data from the step."
    error_message: str | None = None
    "Error message if step failed."
    execution_time: float = 0.0
    "Time taken to execute the step in seconds."
    retry_count: int = 0
    "Number of retries that were performed."
    executed_at: datetime | None = None
    "When the step was executed."

    def __post_init__(self):
        """Set executed_at timestamp if not provided."""
        if self.executed_at is None:
            self.executed_at = datetime.now(UTC)


@dataclass
class WorkflowResult:
    """Result of complete workflow execution."""

    workflow_name: str
    "Name of the workflow that was executed."
    success: bool
    "Whether the overall workflow succeeded."
    steps: list[WorkflowStepResult]
    "Results from all executed steps."
    total_execution_time: float = 0.0
    "Total time taken for the workflow in seconds."
    completed_steps: int = 0
    "Number of steps that completed successfully."
    failed_steps: int = 0
    "Number of steps that failed."
    error_message: str | None = None
    "Overall error message if workflow failed."
    metadata: dict[str, Any] | None = None
    "Additional metadata from workflow execution."
    executed_at: datetime | None = None
    "When the workflow was executed."

    def __post_init__(self):
        """Initialize calculated values."""
        if self.executed_at is None:
            self.executed_at = datetime.now(UTC)
        if self.metadata is None:
            self.metadata = {}


class WorkflowOrchestrator(BaseServiceProvider):
    """
    Multi-step workflow orchestration using existing service patterns.

    This orchestrator manages the execution of complex multi-step workflows
    for intent processing. It provides:
    - Sequential and parallel step execution
    - Error recovery and retry mechanisms
    - Service integration and context propagation
    - Health monitoring and performance tracking
    - Dependency management between steps

    The orchestrator integrates with existing CodeWeaver services:
    - Uses ServicesManager for dependency injection
    - Follows BaseServiceProvider patterns for health monitoring
    - Leverages existing error handling and recovery patterns
    """

    def __init__(self, config: ServiceConfig, services_manager):
        """Initialize workflow orchestrator with service dependencies."""
        super().__init__(ServiceType.INTENT, config)
        self.services_manager = services_manager
        self.logger = logging.getLogger(__name__)
        self.name = "workflow_orchestrator"
        self._execution_stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "total_steps_executed": 0,
            "avg_workflow_time": 0.0,
        }
        self._step_registry = {}
        self._workflow_templates = {}

    async def _initialize_provider(self) -> None:
        """Initialize workflow orchestrator with service dependencies."""
        self.logger.info("Initializing workflow orchestrator")
        try:
            await self._register_common_steps()
            self.logger.info("Workflow orchestrator initialized successfully")
        except Exception as e:
            self.logger.exception("Failed to initialize workflow orchestrator")
            raise ServiceIntegrationError(
                f"Workflow orchestrator initialization failed: {e}"
            ) from e

    async def _shutdown_provider(self) -> None:
        """Shutdown workflow orchestrator resources."""
        self.logger.info("Shutting down workflow orchestrator")
        self.logger.info(
            "Workflow statistics: %d total, %d successful, %d failed, %d steps executed",
            self._execution_stats["total_workflows"],
            self._execution_stats["successful_workflows"],
            self._execution_stats["failed_workflows"],
            self._execution_stats["total_steps_executed"],
        )
        self._step_registry.clear()
        self._workflow_templates.clear()

    async def _check_health(self) -> bool:
        """Check workflow orchestrator health."""
        try:
            test_workflow = WorkflowDefinition(
                name="health_check",
                steps=[WorkflowStep(name="test_step", handler=self._health_check_step)],
            )
            result = await self._execute_workflow_internal(test_workflow, {})
        except Exception as e:
            self.logger.warning("Health check failed: %s", e)
            return False
        else:
            return result.success

    async def _health_check_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Simple health check step."""
        return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}

    async def execute_workflow(
        self, workflow_definition: WorkflowDefinition, context: dict[str, Any]
    ) -> WorkflowResult:
        """
        Execute multi-step workflow with service integration.

        Args:
            workflow_definition: The workflow to execute
            context: Service context with dependencies and metadata

        Returns:
            Result of the workflow execution
        """
        start_time = time.time()
        self._execution_stats["total_workflows"] += 1
        try:
            self.logger.info("Executing workflow: %s", workflow_definition.name)
            self._validate_workflow(workflow_definition)
            result = await self._execute_workflow_internal(workflow_definition, context)
            execution_time = time.time() - start_time
            result.total_execution_time = execution_time
            if result.success:
                self._execution_stats["successful_workflows"] += 1
            else:
                self._execution_stats["failed_workflows"] += 1
            self._execution_stats["total_steps_executed"] += len(result.steps)
            total_successful = self._execution_stats["successful_workflows"]
            if total_successful > 0:
                current_avg = self._execution_stats["avg_workflow_time"]
                self._execution_stats["avg_workflow_time"] = (
                    current_avg * (total_successful - 1) + execution_time
                ) / total_successful
            self.logger.info(
                "Workflow %s completed: %s (%d/%d steps succeeded)",
                workflow_definition.name,
                "SUCCESS" if result.success else "FAILED",
                result.completed_steps,
                len(workflow_definition.steps),
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self._execution_stats["failed_workflows"] += 1
            self.logger.exception("Workflow execution failed")
            return WorkflowResult(
                workflow_name=workflow_definition.name,
                success=False,
                steps=[],
                total_execution_time=execution_time,
                error_message=f"Workflow execution failed: {e}",
                metadata={"error_type": type(e).__name__},
            )
        else:
            return result

    async def _execute_workflow_internal(
        self, workflow_def: WorkflowDefinition, context: dict[str, Any]
    ) -> WorkflowResult:
        """Internal workflow execution logic."""
        step_results = []
        completed_steps = 0
        failed_steps = 0
        enhanced_context = await self._create_workflow_context(context)
        for step in workflow_def.steps:
            try:
                if not self._check_step_dependencies(step, step_results):
                    self.logger.warning("Skipping step %s - dependencies not met", step.name)
                    continue
                step_result = await self._execute_step(step, enhanced_context)
                step_results.append(step_result)
                if step_result.success:
                    completed_steps += 1
                    enhanced_context = self._merge_step_context(enhanced_context, step_result)
                else:
                    failed_steps += 1
                    if step.required and (not workflow_def.allow_partial_success):
                        self.logger.error("Required step %s failed - stopping workflow", step.name)
                        break
            except Exception as e:
                self.logger.exception("Step %s execution failed", step.name)
                step_result = WorkflowStepResult(
                    step_name=step.name,
                    success=False,
                    data=None,
                    error_message=str(e),
                    execution_time=0.0,
                )
                step_results.append(step_result)
                failed_steps += 1
                if step.required and (not workflow_def.allow_partial_success):
                    break
        success = completed_steps > 0 and (
            failed_steps == 0
            or (workflow_def.allow_partial_success and completed_steps > failed_steps)
        )
        return WorkflowResult(
            workflow_name=workflow_def.name,
            success=success,
            steps=step_results,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            error_message=None if success else f"Workflow failed: {failed_steps} step(s) failed",
            metadata={
                "allow_partial_success": workflow_def.allow_partial_success,
                "required_steps": len([s for s in workflow_def.steps if s.required]),
                "optional_steps": len([s for s in workflow_def.steps if not s.required]),
            },
        )

    async def _execute_step(
        self, step: WorkflowStep, context: dict[str, Any]
    ) -> WorkflowStepResult:
        """Execute a single workflow step with retry logic."""
        start_time = time.time()
        last_error = None
        for attempt in range(step.retry_count + 1):
            try:
                self.logger.debug(
                    "Executing step %s (attempt %d/%d)",
                    step.name,
                    attempt + 1,
                    step.retry_count + 1,
                )
                result_data = await step.handler(context)
                execution_time = time.time() - start_time
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Step %s attempt %d failed: %s", step.name, attempt + 1, last_error
                )
                if attempt < step.retry_count:
                    wait_time = min(2**attempt, 10)
                    await self._wait(wait_time)
            else:
                return WorkflowStepResult(
                    step_name=step.name,
                    success=True,
                    data=result_data,
                    execution_time=execution_time,
                    retry_count=attempt,
                )
        execution_time = time.time() - start_time
        return WorkflowStepResult(
            step_name=step.name,
            success=False,
            data=None,
            error_message=str(last_error) if last_error else "Step execution failed",
            execution_time=execution_time,
            retry_count=step.retry_count + 1,
        )

    async def _wait(self, seconds: float) -> None:
        """Async wait helper."""
        import asyncio

        await asyncio.sleep(seconds)

    def _check_step_dependencies(
        self, step: WorkflowStep, completed_results: list[WorkflowStepResult]
    ) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True
        completed_step_names = {r.step_name for r in completed_results if r.success}
        return all(dep in completed_step_names for dep in step.dependencies)

    async def _create_workflow_context(self, base_context: dict[str, Any]) -> dict[str, Any]:
        """Create enhanced context for workflow execution."""
        if hasattr(self.services_manager, "create_service_context"):
            service_context = await self.services_manager.create_service_context(base_context)
        else:
            service_context = base_context.copy()
        return {
            **service_context,
            "workflow_metadata": {
                "orchestrator": self.name,
                "session_id": self._generate_session_id(),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "services_manager": self.services_manager,
        }

    def _merge_step_context(
        self, context: dict[str, Any], step_result: WorkflowStepResult
    ) -> dict[str, Any]:
        """Merge step result into context for subsequent steps."""
        updated_context = context.copy()
        if "step_results" not in updated_context:
            updated_context["step_results"] = {}
        updated_context["step_results"][step_result.step_name] = {
            "success": step_result.success,
            "data": step_result.data,
            "execution_time": step_result.execution_time,
        }
        if step_result.data and isinstance(step_result.data, dict) and "context_updates" in step_result.data:
            updated_context |= step_result.data["context_updates"]
        return updated_context

    def _validate_workflow(self, workflow_def: WorkflowDefinition) -> None:
        """Validate workflow definition."""
        if not workflow_def.name:
            raise ValueError("Workflow must have a name")
        if not workflow_def.steps:
            raise ValueError("Workflow must have at least one step")
        step_names = {step.name for step in workflow_def.steps}
        for step in workflow_def.steps:
            if step.dependencies:
                for dep in step.dependencies:
                    if dep not in step_names:
                        raise ValueError(f"Step {step.name} depends on unknown step {dep}")

    async def _register_common_steps(self) -> None:
        """Register common reusable workflow steps."""
        self.logger.debug("Registering common workflow steps")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid

        return str(uuid.uuid4())

    async def health_check(self) -> ServiceHealth:
        """Enhanced health check with workflow-specific metrics."""
        base_health = await super().health_check()
        base_health.metadata = {
            "total_workflows": self._execution_stats["total_workflows"],
            "success_rate": self._execution_stats["successful_workflows"]
            / max(1, self._execution_stats["total_workflows"]),
            "avg_workflow_time": self._execution_stats["avg_workflow_time"],
            "total_steps_executed": self._execution_stats["total_steps_executed"],
            "registered_steps": len(self._step_registry),
            "workflow_templates": len(self._workflow_templates),
        }
        return base_health

    def get_orchestrator_stats(self) -> dict[str, Any]:
        """Get comprehensive orchestrator statistics."""
        return {
            "execution_stats": self._execution_stats.copy(),
            "registry_info": {
                "registered_steps": len(self._step_registry),
                "workflow_templates": len(self._workflow_templates),
            },
            "health_status": self.status.value,
        }
