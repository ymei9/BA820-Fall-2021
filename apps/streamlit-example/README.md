# About

Basic example of streamlit.  


## Dockerfile

If you want to build a docker image, you need to ensure that you have Docker installed and the engine running.  T

Build an image locally with

`docker build -f Dockerfile -t sms-spam-app .`

Above will create a docker app via the config of the `Dockerfile`, which basically grabs all of the files in the local directory, and moves them to the docker container.

From there, the docker container knows to launch streamlit and the `app.py` file which represents the app.

To view your images:

```
docker images
```

In my case, I currently see this:

```
(ba820) Brocks-MBP-2:streamlit-example btibert$ docker images
REPOSITORY                 TAG       IMAGE ID       CREATED         SIZE
sms-spam-app               latest    882fb43edf61   2 minutes ago   911MB
heartexlabs/label-studio   latest    abb24d7f26e3   2 weeks ago     1.26GB
```

To run an image, detached (that is , not active in the terminal)

```
docker run -d -p 8501:8501 sms-spam-app
```

You might see something __similar__ to this:

```
(ba820) Brocks-MBP-2:streamlit-example btibert$ docker run -d -p 8501:8501 sms-spam-app
68be8d04dd9564dcaf58dfe31d14a0504292c2f290bc9d6e0e4011f2ef730b3d
```

From here, we can navigate to `localhost:8501` __in a browser on our laptop__.  If you are running this in the cloud, you would go to the public IP of your machine, followed by `:8501`.  Refer to the SMS challege URL for an example.

To `stop` the app, you will need to tell docker to stop the Container id.  Because our app is running, you can get this via `docker ps`

```
(ba820) Brocks-MBP-2:streamlit-example btibert$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED         STATUS         PORTS                    NAMES
68be8d04dd95   sms-spam-app   "streamlit run app.py"   3 minutes ago   Up 3 minutes   0.0.0.0:8501->8501/tcp   angry_dewdney
```

To stop:

```
docker stop 68be8d04dd95
```

Where `68be8d04dd95` is the Docker id listed via `docker ps`.

That's it!

> You _could_ manage this via the Docker UI tool, but I feel like there is more control when using the command line/bash/terminal.

