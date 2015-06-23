# Flask Routes

*Generated Jun 23, 2015*

Documentation on the various routes used internally for the web application.

These are all technically RESTful, but they return HTML pages. To get JSON responses, see [this page](API.md).

### Table of Contents

* [Home](#home)
* [Jobs](#jobs)
* [Datasets](#datasets)
* [Models](#models)
* [Util](#util)

## Home

### `/`

> DIGITS home page

> Returns information about each job on the server

> 

> Returns JSON when requested:

> {

> datasets: [{id, name, status},...],

> models: [{id, name, status},...]

> }

Methods: **GET**

Location: [`digits/views.py@21`](../digits/views.py#L21)

## Jobs

### `/datasets/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@147`](../digits/views.py#L147)

### `/datasets/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@167`](../digits/views.py#L167)

### `/datasets/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@128`](../digits/views.py#L128)

### `/jobs/<job_id>`

> Redirects to the appropriate /datasets/ or /models/ page

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@95`](../digits/views.py#L95)

### `/jobs/<job_id>`

> Edit the name of a job

Methods: **PUT**

Arguments: `job_id`

Location: [`digits/views.py@112`](../digits/views.py#L112)

### `/jobs/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@147`](../digits/views.py#L147)

### `/jobs/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@167`](../digits/views.py#L167)

### `/jobs/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@128`](../digits/views.py#L128)

### `/models/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@147`](../digits/views.py#L147)

### `/models/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@167`](../digits/views.py#L167)

### `/models/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@128`](../digits/views.py#L128)

## Datasets

### `/datasets/<job_id>`

> Show a DatasetJob

> 

> Returns JSON when requested:

> {id, name, directory, status}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/dataset/views.py@15`](../digits/dataset/views.py#L15)

### `/datasets/images/classification`

> Creates a new ImageClassificationDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/classification/views.py@217`](../digits/dataset/images/classification/views.py#L217)

### `/datasets/images/classification/new`

> Returns a form for a new ImageClassificationDatasetJob

Methods: **GET**

Location: [`digits/dataset/images/classification/views.py@207`](../digits/dataset/images/classification/views.py#L207)

### `/datasets/images/regression`

> Creates a new ImageClassificationDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/regression/views.py@167`](../digits/dataset/images/regression/views.py#L167)

### `/datasets/images/regression/new`

> Returns a form for a new ImageRegressionJob

Methods: **GET**

Location: [`digits/dataset/images/regression/views.py@157`](../digits/dataset/images/regression/views.py#L157)

### `/datasets/images/resize-example`

> Resizes the example image, and returns it as a string of png data

Methods: **POST**

Location: [`digits/dataset/images/views.py@18`](../digits/dataset/images/views.py#L18)

### `/datasets/summary`

> Return a short HTML summary of a DatasetJob

Methods: **GET**

Location: [`digits/dataset/views.py@38`](../digits/dataset/views.py#L38)

## Models

### `/models/<job_id>`

> Show a ModelJob

> 

> Returns JSON when requested:

> {id, name, directory, status, snapshots: [epoch,epoch,...]}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/model/views.py@27`](../digits/model/views.py#L27)

### `/models/<job_id>/download`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension` (`tar.gz`)

Location: [`digits/model/views.py@167`](../digits/model/views.py#L167)

### `/models/<job_id>/download.<extension>`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension`

Location: [`digits/model/views.py@167`](../digits/model/views.py#L167)

### `/models/customize`

> Returns a customized file for the ModelJob based on completed form fields

Methods: **POST**

Location: [`digits/model/views.py@51`](../digits/model/views.py#L51)

### `/models/images/classification`

> Create a new ImageClassificationModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/classification/views.py@51`](../digits/model/images/classification/views.py#L51)

### `/models/images/classification/classify_many`

> Classify many images and return the top 5 classifications for each

> 

> Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}

Methods: **GET**, **POST**

Location: [`digits/model/images/classification/views.py@289`](../digits/model/images/classification/views.py#L289)

### `/models/images/classification/classify_one`

> Classify one image and return the top 5 classifications

> 

> Returns JSON when requested: {predictions: {category: confidence,...}}

Methods: **GET**, **POST**

Location: [`digits/model/images/classification/views.py@233`](../digits/model/images/classification/views.py#L233)

### `/models/images/classification/large_graph`

> Show the loss/accuracy graph, but bigger

Methods: **GET**

Location: [`digits/model/images/classification/views.py@220`](../digits/model/images/classification/views.py#L220)

### `/models/images/classification/new`

> Return a form for a new ImageClassificationModelJob

Methods: **GET**

Location: [`digits/model/images/classification/views.py@30`](../digits/model/images/classification/views.py#L30)

### `/models/images/classification/top_n`

> Classify many images and show the top N images per category by confidence

Methods: **POST**

Location: [`digits/model/images/classification/views.py@366`](../digits/model/images/classification/views.py#L366)

### `/models/images/regression`

> Create a new ImageClassificationModelJob

Methods: **POST**

Location: [`digits/model/images/regression/views.py@53`](../digits/model/images/regression/views.py#L53)

### `/models/images/regression/classify_many`

> Classify many images and return the top 5 classifications for each

Methods: **POST**

Location: [`digits/model/images/regression/views.py@279`](../digits/model/images/regression/views.py#L279)

### `/models/images/regression/classify_one`

> Classify one image and return the predictions, weights and activations

Methods: **POST**

Location: [`digits/model/images/regression/views.py@228`](../digits/model/images/regression/views.py#L228)

### `/models/images/regression/large_graph`

> Show the loss/accuracy graph, but bigger

Methods: **GET**

Location: [`digits/model/images/regression/views.py@216`](../digits/model/images/regression/views.py#L216)

### `/models/images/regression/new`

> Return a form for a new ImageClassificationModelJob

Methods: **GET**

Location: [`digits/model/images/regression/views.py@32`](../digits/model/images/regression/views.py#L32)

### `/models/images/regression/top_n`

> Classify many images and show the top N images per category by confidence

Methods: **POST**

Location: [`digits/model/images/regression/views.py@349`](../digits/model/images/regression/views.py#L349)

### `/models/visualize-lr`

> Returns a JSON object of data used to create the learning rate graph

Methods: **POST**

Location: [`digits/model/views.py@112`](../digits/model/views.py#L112)

### `/models/visualize-network`

> Returns a visualization of the custom network as a string of PNG data

Methods: **POST**

Location: [`digits/model/views.py@99`](../digits/model/views.py#L99)

## Util

### `/files/<path:path>`

> Return a file in the jobs directory

> 

> If you install the nginx.site file, nginx will serve files instead

> and this path will never be used

Methods: **GET**

Arguments: `path`

Location: [`digits/views.py@211`](../digits/views.py#L211)

