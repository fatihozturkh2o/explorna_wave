# Running ExploRNA

Clone the repo to your local.

Before everything make sure that wave is ready:
Download and extract [Wave](https://github.com/h2oai/wave/releases/tag/v0.10.0). Start the server with

```bash
./waved
```

**1.** Download and install viennaRNA based on your operating system at middle in the page here: https://www.tbi.univie.ac.at/RNA/#download

**Imporant**: If you use MacOS as operating system, make sure that you see RNA related files in the directory of <code>/usr/local/bin</code> after installing the proper package.

If you use Ubuntu, after downloading and installing the appropriate package, RNA related files most probably will be located at <code>/usr/bin</code>. So, please change the line: <code>echo "vienna_2: /usr/local/bin" > arnie.conf;</code>  with <code>echo "vienna_2: /usr/bin" > arnie.conf;</code> in setup_draw_rna.sh before running it.

**2.** Open terminal in the cloned folder and run: <code>make setup</code>

**3.** Then run the bash file: for Ubuntu: <code>bash Ubuntu_setup_draw_rna.sh</code> , for Mac: <code>bash MacOS_setup_draw_rna.sh</code>

**4.** Now you are ready to run the app: <code>./venv/bin/python run.py</code>   
