"""Local MCP transport check; no Modal credentials or cloud calls."""
import asyncio
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

@unittest.skipUnless(importlib.util.find_spec('mcp'), 'MCP dependency not installed')
class MCPTests(unittest.TestCase):
    def test_stdio_status_and_no_execute_tool(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        async def check():
            with tempfile.TemporaryDirectory() as state:
                params = StdioServerParameters(command=sys.executable, args=['-m','modal_control','--state',state,'mcp'])
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=45) as session:
                        initialized = await session.initialize()
                        self.assertIn("single explicit user release", initialized.instructions)
                        result = await session.list_tools()
                        names = {tool.name for tool in result.tools}
                        self.assertEqual(names, {'get_status','prepare_run_plan','get_run_plan','get_run','get_run_events','prepare_batch_plan','get_batch_plan'})
                        response = await session.call_tool('get_status', {})
                        self.assertFalse(response.is_error)
        asyncio.run(check())
