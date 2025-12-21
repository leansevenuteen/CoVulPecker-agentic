# Joern Installation Guide for macOS

This guide documents the installation of Joern on macOS for code analysis and vulnerability detection.

## Installation Steps

### 1. Install coreutils

```bash
brew install coreutils
```

### 2. Clone Joern repository

```bash
git clone https://github.com/joernio/joern.git
cd joern
```

### 3. Build Joern

```bash
sbt stage
```

### 4. Install Java JDK 19

```bash
brew install openjdk@19
```

## Installation Complete

Joern is now installed and ready to use for generating Code Property Graphs (CPGs) from source code.
