"""Plan/inspect-only MCP surface for human-approved Modal batches."""
from .plans import prepare_plan, show_plan, prepare_batch_plan as prepare_batch, show_batch_plan as show_batch


def create_server(store, catalog_path):
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations
    server = MCPServer(
        'modal-control', title='Modal Control', version='0.2.0',
        instructions=(
            'All Modal work must use this controller. A single explicit user release can '
            'authorize the entire bounded batch, its dependent steps and its reviewed '
            'retry limits; do not ask again for steps inside that authorized scope. '
            'Preserve the actual user message and disclose model, inputs, time, cost, '
            'and concurrency. Changes outside that scope need new authorization. '
            'This MCP is plan/inspect only; the agent uses the controller CLI to execute. '
            'T4-only GPU and CPU allocations remain bounded; never hide allocations.'
        ),
    )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def get_status() -> dict:
        """Read recorded runs and alerts without contacting Modal."""
        return store.overview()

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
    def prepare_run_plan(job: str, catalog_file: str = "") -> dict:
        """Prepare an expiring exact plan locally. This never launches a job."""
        return prepare_plan(store, catalog_file or catalog_path, job)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
    def prepare_batch_plan(spec_file: str) -> dict:
        """Prepare one bounded batch plan locally without starting any job."""
        return prepare_batch(store, spec_file)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def get_batch_plan(plan_id: str) -> dict:
        """Read and verify a prepared batch plan."""
        return show_batch(store, plan_id)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def get_run_plan(plan_id: str) -> dict:
        """Read and integrity-check a pending plan for user review."""
        return show_plan(store, plan_id)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def get_run(run_id: str) -> dict:
        """Read the saved run manifest and status."""
        return store.get(run_id)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def get_run_events(run_id: str, after: int = 0) -> list:
        """Read saved progress and stdout/stderr events."""
        return store.events(run_id, after)

    return server
