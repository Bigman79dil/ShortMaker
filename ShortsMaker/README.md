Made by: Bigman79dil



What you need



Python 3.10+

FFmpeg (installed and added to your system Path)

Ollama (with `mistral-small3.2` model installed)



Install



Open PowerShell as an Administrator, paste the following command, and press 'Enter'. It will check your system and only install the software or Python packages you are missing:



\-----------------------------------------------------------------------------------------------------------



```powershell

\# Install missing system tools via Winget

if (!(Get-Command python -ErrorAction SilentlyContinue)) { echo "Installing Python..."; winget install Python.Python.3.11 --silent }

if (!(Get-Command ffmpeg -ErrorAction SilentlyContinue)) { echo "Installing FFmpeg..."; winget install Gyan.FFmpeg --silent }

if (!(Get-Command ollama -ErrorAction SilentlyContinue)) { echo "Installing Ollama..."; winget install Ollama.Ollama --silent }



\------------------------------------------------------------------------------------------------------------

MAKE SURE YOU REOPEN POWERSHELL(as an admin)! then paste this into it



\# Pull the AI model if Ollama is running

\& ollama pull mistral-small3.2



\# 3. Install required Python packages

python -m pip install --upgrade pip

python -m pip install openai-whisper ollama ffmpeg-python

```


Instructions



1\. Put your long video inside the folder and name it `input.mp4`.

2\. Double-click `run.bat` to start it (takes a few mins to finish depending on your pc specs)

3\. Check the `output\_shorts` folder for your finished clips.

4\. Double-click `clear.bat` whenever you want to delete the generated clips and reset.



