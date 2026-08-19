def start_camera(camera):
    camera.start()
    return {"status": "starting", "camera_status": camera.status}


def stop_camera(camera):
    camera.stop()
    return {"status": "stopped", "camera_status": camera.status}


def configure_camera(camera, url: str):
    if not url.startswith(("http://", "https://")):
        raise ValueError("camera URL must use http:// or https://")
    camera.stop()
    camera.url = url
    camera.start()
    return {"status": "configured", "camera_status": camera.status}
