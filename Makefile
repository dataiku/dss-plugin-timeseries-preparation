PLUGIN_VERSION=1.1.3
PLUGIN_ID=timeseries-preparation

plugin:
	cat plugin.json|json_pp > /dev/null
	rm -rf dist
	mkdir dist
	zip --exclude "*.pyc" -r dist/dss-plugin-${PLUGIN_ID}-${PLUGIN_VERSION}.zip plugin.json code-env custom-recipes python-lib
