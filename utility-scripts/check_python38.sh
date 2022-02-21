#!/usr/bin/env bash

if [ -z $(which python) ]
then
    echo "No python found (which python). You need python 3.8."
    exit 125
else
    PYTHON_V=$(python -V)
    if [[ $PYTHON_V != *"Python 3.8"* ]]
    then
        echo "To build, your default python must be version 3.8. Your default is configured to $(python -V 2>&1)"
        exit 125
    fi
fi
