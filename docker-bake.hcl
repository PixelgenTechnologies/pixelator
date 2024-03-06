variable export_cache {
    default = false
}

group default {
    targets = ["pixelator-prod"]
}

variable github_registry_cache {
    default = "ghcr.io/pixelgentechnologies/pixelator:buildcache"
}

variable aws_registry_cache {
    default = "890888997283.dkr.ecr.eu-north-1.amazonaws.com/pixelator:buildcache"
}

variable BUILD_DATE {
    default = timestamp()
}

variable VCS_REF {
    default = null
}

variable BUILD_VERSION {
    default = null
}

target "pixelator-base" {
    dockerfile = "containers/base.Dockerfile"
    cache-from = [
        "type=registry,ref=${aws_registry_cache}",
        "type=registry,ref=${github_registry_cache}",
    ]
    cache-to = export_cache ? [
        "type=registry,ref=${aws_registry_cache}",
        "type=registry,ref=${github_registry_cache}",
    ] : []

    output = [
        "type=docker"
    ]
}



target pixelator-dev {
    depends_on = ["pixelator-base"]
    contexts = {
        pixelator-base = "target:pixelator-base"
    }
    dockerfile = "containers/dev.Dockerfile"
    tags = [ "pixelator:latest-dev" ]
}


target pixelator-prod {
    depends_on = ["pixelator-base"]
    contexts = {
        pixelator-base = "target:pixelator-base"
    }
    dockerfile = "containers/prod.Dockerfile"
    tags = [ "pixelator:latest" ]
    args = {
        BUILD_DATE = BUILD_DATE
        VCS_REF = VCS_REF
        BUILD_VERSION = BUILD_VERSION
    }
}
