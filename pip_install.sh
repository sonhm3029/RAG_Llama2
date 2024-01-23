#!/bin/bash

# Check if a package name is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <package_name>"
    exit 1
fi

# Specify the package name from the command-line argument
package_name="$1"

# Install the package and get its version from pip show
package_version=$(pip show $package_name | awk '/^Version:/ {print $2}')

# Check if the package is already in requirements.txt
if grep -q "$package_name==$package_version" requirements.txt; then
    echo "Package $package_name==$package_version is already in requirements.txt"
else
    # Append the package and version to requirements.txt
    echo "$package_name==$package_version" >> requirements.txt
    echo "Added $package_name==$package_version to requirements.txt"
fi
