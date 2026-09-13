FROM node:20-bookworm-slim

RUN npm install --global --no-fund --no-audit mcp-echarts@0.7.1

ENTRYPOINT ["mcp-echarts"]
CMD ["-t", "streamable"]
