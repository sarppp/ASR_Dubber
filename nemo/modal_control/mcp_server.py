"""Plan/inspect-only MCP surface; execution requires the user's explicit go."""
from .plans import prepare_plan, show_plan


def create_server(store, catalog_path):
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations
    server = MCPServer(
        'modal-control', title='Modal Control', version='0.2.0',
        instructions=(
            'All Modal work must use this controller. Prepare and explain an exact plan, '
            'then wait for the human to explicitly say release it for that plan. '
            'This MCP has no execution tool. After that explicit go, the agent may '
            'invoke the controller execute CLI with the exact plan id and the user approval '
            'message. Never self-approve, call Modal directly, or silently change the plan. '
            'GPU requests are T4-only; CPU is allowed. No fallback or automatic retries.'
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
