---
title: Image similarity search web_demo
description: Image similarity search demo running as a Flask web server.
category: example
include_in_docs: true
priority: 10
---

# Web Demo

## Requirements

The demo server requires Python with some dependencies.
To make sure you have the dependencies, please run `pip install -r examples/web_demo/requirements.txt`.


## Run

Running `python3 app.py` will bring up the demo server, accessible at `http://0.0.0.0:8080`.
You can enable debug mode of the web server, or switch to a different port:

    % python3 app.py -h
    Usage: app.py [options]

    Options:
      -h, --help            show this help message and exit
      -d, --debug           enable debug mode
      -p PORT, --port=PORT  which port to serve content on
