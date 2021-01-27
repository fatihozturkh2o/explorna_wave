# Running ExploRNA

Clone the repo to your local.

Before everything make sure that wave is ready:
Download and extract [Wave](https://github.com/h2oai/wave/releases/tag/v0.10.0). Start the server with

```bash
./waved
```

**1.** Download and install viennaRNA based on your operating system at middle in the page here: https://www.tbi.univie.ac.at/RNA/#download

**Imporant**: If you use MacOS as operating system, make sure that you see RNA related files in the directory of <code>/usr/local/bin</code> after installing the proper package. For Ubuntu, they will be located at usr/bin/. This is why there are two different bash files.

**2.** Open terminal in the cloned folder and run: <code>make setup</code>

**3.** Then run the bash file: for Ubuntu: <code>bash Ubuntu_setup_draw_rna.sh</code> , for Mac: <code>bash MacOS_setup_draw_rna.sh</code>

**4.** Now you are ready to run the app: <code>./venv/bin/python run.py</code>   
