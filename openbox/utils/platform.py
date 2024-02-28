
def get_platform():
    import platform
    os_name = platform.system()
    if os_name == "Linux":
        return "Linux"
    elif os_name == "Darwin":
        return "OSX"
    elif os_name == "Windows":
        return "Windows"
    else:
        return "Unknown"
