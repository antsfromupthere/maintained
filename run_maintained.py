# from alpha_agent.NetworkOptimizationTool import NetworkOptimizationTool
from maintained.agent_tool.GeographicDataCollectionTool import GeographicDataCollectionTool
from maintained.agent_tool.NetworkAnalysisTool import NetworkAnalysisTool
from maintained.agent_tool.NetOptTool import NetworkOptimizationTool

from smolagents import Tool, ToolCallingAgent, LiteLLMModel

# Create the network deployment agent

model_id="ollama_chat/qwen3:4b"
api_base="http://127.0.0.1:11434"
model = LiteLLMModel(
    model_id=model_id,
    api_base=api_base,
    num_ctx=32000,
)

agent = ToolCallingAgent(
    tools=[
        GeographicDataCollectionTool(),
        NetworkAnalysisTool(),
        NetworkOptimizationTool(),
    ],
    model=model
)
agent.run(
"""
I'm working on a plan to set up a wireless network in a region of Saudi Arabia, specifically within the area between 21.0 and 21.5 degrees latitude, and 43.5 to 44.0 degrees longitude. The network will operate on the 5 GHz frequency band, using a 10 MHz channel bandwidth.

For the infrastructure, I'm considering a combination of High Altitude Platforms (HAPs) and Terrestrial Base Stations (TBSs). Each HAP is estimated to cost around 1200 units, and each TBS about 600 units. Both types of stations will transmit at 20 watts of power.

Could you help analyze this area for me? I’m particularly interested in understanding how much the total deployment would cost, what kind of average data rates customers can expect, and how many HAPs and TBSs we’ll need to deploy. It would also be great to know where these should be placed, and whether all demand nodes can meet the minimum requirement of 2 Mbps.
"""
)