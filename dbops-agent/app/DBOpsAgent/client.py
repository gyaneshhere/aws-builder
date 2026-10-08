"""Invoke a DBOps AgentCore Runtime session."""
from __future__ import annotations
import argparse, json
from typing import Any
from aws_bedrock_token_generator import provide_token
from openai import NotFoundError, OpenAI
BMA_MODEL_ID="openai.gpt-5.6-luna"
WORKSPACE_DIRECTORY="/mnt/home/workspace"
CAPABILITY_DIRECTORIES=["/opt/bma/plugins"]
TURN_END=("completed","failed","cancelled")
TOOL_CALLS=("mcp_call","function_call","web_search_call")
INSTRUCTIONS="""You are DBOps Agent, a read-only database incident investigator. Never make production changes. Use installed dbops skills and diagnostic commands. Never invent metrics or claim a recommended action was executed. Every conclusion must be evidence-backed with confidence. Always state that no production changes were performed."""

def show(data:dict[str,Any])->None:
    kind=data["type"].removeprefix("agent.session."); item=data.get("item") or {}
    if kind=="created": print(f"Session {data['session']['id']}")
    elif kind=="turn.output_text.delta": print(data["delta"],end="",flush=True)
    elif kind=="turn.item.done" and item.get("type")=="command_execution": print(f"\n$ {item['command']}\n{item.get('output') or ''}".rstrip())
    elif kind=="turn.item.done" and item.get("type") in TOOL_CALLS: print(f"\nTool {item.get('name') or item['type']} {item.get('status')}")
    elif kind=="error" or kind.split(".")[-1] in TURN_END:
        source=data.get("turn") or data.get("environment") or data.get("session"); print(f"\n{kind} {(source or data).get('error') or ''}".rstrip())

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--runtime",required=True); p.add_argument("--session-id"); p.add_argument("--input",required=True); p.add_argument("--gateway"); p.add_argument("--raw",action="store_true"); p.add_argument("--delete",action="store_true"); a=p.parse_args()
    region=a.runtime.split(":")[3]
    with OpenAI(api_key=lambda:provide_token(region=region),base_url=f"https://bedrock-mantle.{region}.api.aws/openai/v1") as client:
        sessions=client.beta.agents.sessions; sid=a.session_id
        if sid:
            try: session=sessions.retrieve(sid).model_dump(warnings=False)
            except NotFoundError: sid=None
            else:
                if session["environment"].get("runtime_arn")!=a.runtime: raise ValueError("Session belongs to a different Runtime")
        if sid:
            events=sessions.events.stream(sid,extra_query={"stream":"true"}); msg={"role":"user","content":[{"type":"input_text","text":a.input}]}; sessions.events.create(sid,events=[{"type":"agent.session.input.message","input":[msg]}])
        else:
            agent={"model":BMA_MODEL_ID,"instructions":INSTRUCTIONS}
            if a.gateway: agent["tools"]=[{"type":"mcp","server_label":"dbops_tools","required":True,"connection_origin":"service","transport":{"type":"http","server_url":a.gateway}}]
            events=sessions.create(agent=agent,environment={"type":"aws_bedrock_agentcore","runtime_arn":a.runtime,"runtime_qualifier":"DEFAULT","workspace_directory":WORKSPACE_DIRECTORY,"capability_directories":CAPABILITY_DIRECTORIES},input=a.input,stream=True)
        with events:
            for event in events:
                data=event.model_dump(mode="json",warnings=False); sid=sid or (data.get("session") or {}).get("id")
                print(json.dumps(data) if a.raw else "",end="" if a.raw else "") if a.raw else show(data)
                if data["type"].removeprefix("agent.session.turn.") in TURN_END: break
        if a.delete and sid: sessions.delete(sid); print(f"\nDeleted session {sid}")
if __name__=="__main__": main()
