from dify_plugin import Plugin, DifyPluginEnv
import os
import yaml

# Load and patch manifest to handle missing icon and create required directories
def patch_manifest():
    try:
        # Create _assets directory if it doesn't exist
        if not os.path.exists('_assets'):
            os.makedirs('_assets')
            
        # Create a minimal icon in _assets
        icon_content = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6366F1" stroke-width="2">
  <path d="M9 19c-5 0-8-3-8-6s3-6 8-6c2 0 4 1 5.5 2.5L21 3"/>
  <path d="m21 3-1 6-6-1"/>
  <path d="M9 10h.01"/>
  <path d="M9 13h.01"/>
  <path d="M9 16h.01"/>
</svg>'''
        with open('_assets/icon.svg', 'w') as f:
            f.write(icon_content)
            
        with open('manifest.yaml', 'r') as f:
            manifest = yaml.safe_load(f)
        
        # Ensure icon field points to the correct location
        manifest['icon'] = 'icon.svg'
            
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
            tool_config['identity']['icon'] = 'icon.svg'
                
        with open('llamaindex_rag.yaml', 'w') as f:
            yaml.dump(tool_config, f, default_flow_style=False)
            
    except Exception:
        pass  # Ignore errors, use original files

patch_manifest()
plugin = Plugin(DifyPluginEnv())

if __name__ == "__main__":
    plugin.run()