from dify_plugin import Plugin, DifyPluginEnv
import os
import yaml

# Load and patch manifest to handle missing icon
def patch_manifest():
    try:
        with open('manifest.yaml', 'r') as f:
            manifest = yaml.safe_load(f)
        
        # Add empty icon if not present to satisfy validation
        if 'icon' not in manifest:
            manifest['icon'] = ""
            
        with open('manifest.yaml', 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
            
        # Also patch the tool YAML file
        with open('llamaindex_rag.yaml', 'r') as f:
            tool_config = yaml.safe_load(f)
            
        # Ensure identity has required fields
        if 'identity' in tool_config:
            if 'description' not in tool_config['identity']:
                tool_config['identity']['description'] = {
                    'en_US': 'Advanced RAG tool for document querying',
                    'zh_Hans': '文档查询的高级RAG工具'
                }
            if 'icon' not in tool_config['identity']:
                tool_config['identity']['icon'] = ""
                
        with open('llamaindex_rag.yaml', 'w') as f:
            yaml.dump(tool_config, f, default_flow_style=False)
            
    except Exception:
        pass  # Ignore errors, use original files

patch_manifest()
plugin = Plugin(DifyPluginEnv())

if __name__ == "__main__":
    plugin.run()