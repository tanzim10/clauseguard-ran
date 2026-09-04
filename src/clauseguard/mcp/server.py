"""MCP server placeholder (Week 11). Lists tools; no real behavior."""

TOOL_NAMES = (
    "search_o_ran_specs",
    "run_rca",
    "summarize_kpi_window",
)


def main() -> None:
    print("ClauseGuardRAN MCP server (scaffold stub)")
    print("Planned tools:")
    for name in TOOL_NAMES:
        print(f"  - {name}")
    print("TODO: Week 11 — implement MCP tools (not implemented yet)")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
