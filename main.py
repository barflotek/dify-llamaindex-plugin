from dify_plugin import Plugin, DifyPluginEnv
import os
import yaml

# Simple plugin initialization without any icon handling
def patch_manifest():
    try:
        # Just ensure basic structure without icons
        with open('llamaindex_rag.yaml', 'r') as f:
            tool_config = yaml.safe_load(f)
            
        # Ensure identity has required fields (except icon)
        if 'identity' in tool_config:
            if 'description' not in tool_config['identity']:
                tool_config['identity']['description'] = {
                    'en_US': 'Advanced RAG tool for document querying',
                    'zh_Hans': '文档查询的高级RAG工具'
                }
                
        with open('llamaindex_rag.yaml', 'w') as f:
            yaml.dump(tool_config, f, default_flow_style=False)
            
    except Exception:
        pass  # Ignore errors, use original files

patch_manifest()
plugin = Plugin(DifyPluginEnv())

if __name__ == "__main__":
    plugin.run()