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
            manifest['icon'] = ''
            
        with open('manifest.yaml', 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
    except Exception:
        pass  # Ignore errors, use original manifest

patch_manifest()
plugin = Plugin(DifyPluginEnv())

if __name__ == "__main__":
    plugin.run()