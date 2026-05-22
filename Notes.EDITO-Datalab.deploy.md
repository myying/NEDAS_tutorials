# Custom docker image for the tutorial environment

Use "add-your-service" in Process to create a service using custom image

- requires support to open permission for this.
- Configuration: Container Image: specify image tag (dockerhub tag)
- Configuration: Port: specify port (matching -p option in jupyter service)
- Metadata: name of the service (will be shown in catalog); version (a new version everytime there is changes in the image)
- environment variables: JUPYTER_TOKEN:"" to enable password-less jupyter client

When the add-your-service is run, a few minutes, the new service will appear in Service catalog > Playground

# Deploy the service to official catalog

## Extract the service chart

- launch a Jupyter-python service with "edit role": `https://datalab.dive.edito.eu/launcher/ide/jupyter-python?name=jupyter-python-edit-role&shared=false&autoLaunch=true&kubernetes.role=%C2%ABedit%C2%BB`
- in the terminal: `helm repo add service-playground https://gitlab.mercator-ocean.fr/api/v4/projects/1974/packages/helm/service-playground`
- and `helm pull service-playground/nedas-tutorials --untar` to extract the nedas-tutorials directory

## Merge changes to Service Helm Charts

- requires support to open permission to the `https://gitlab.mercator-ocean.fr/pub/edito-infra/service-helm-charts` repo
- create a new branch
- put the nedas-tutorials directory under "ocean-modelling" category
- run `pre-commit run --all-files` to clean up format
- commit changes and push
- create merge request, then after approval it will be shown in the Service Catalog

# Deploy the tutorial page details

- requires support to open permission to the `https://gitlab.mercator-ocean.fr/pub/edito-infra/edito-tutorials-content` repo
- create a new branch
- add entry of the tutorial in `tutorials.json`
- deploymentUrl: the Launch link
- category: Ocean modelling
- and other metadata


