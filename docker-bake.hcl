variable export_cache {
    default = false
}

group default {
    targets = ["pixelator-prod"]
}

variable github_registry {
    default = "ghcr.io/pixelgentechnologies/pixelator:buildcache"
}

variable aws_registry {
    default = "890888997283.dkr.ecr.eu-north-1.amazonaws.com/pixelator:buildcache"
}

target "pixelator-base" {
    dockerfile = "containers/base.Dockerfile"
    cache-from = [
        "type=registry,ref=${aws_registry}",
        "type=registry,ref=${github_registry}",
    ]
    cache-to = export_cache ? [
        "type=registry,ref=${aws_registry}",
        "type=registry,ref=${github_registry}",
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
}
