### How to use Sphinx

1. Create a directory called `docs`
2. Open the directory `/usr/local/Cellar/sphinx-doc/1.7.5_1/bin` and drag the `quickstart` thing into a terminal which is `cd docs`
3. Follow the prompts accepting everything default and adding the option to auto doc files.
4. Add the path of your scripts to the conf.py file:
```python
import sys
sys.path.append('../../heat/')
```
5. In terminal run `make html`
6. Open and view the html `open build/html/index.html`
7. To run the auto doc go `sphinx-apidoc -o source/ ../heat` where source is where the .rst file is stored and mylib is my library
8. Remake the html and open to view it. It may be necessary to add the modules.rst file as a line modules in the Toc of index.rst:
```python

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

```
