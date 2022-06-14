presentation: pymor.ipynb venv qr_school_website.png qr_docs.png qr_self.png
	. venv/bin/activate ; jupyter notebook pymor.md

pymor.ipynb: venv
	. venv/bin/activate && jupytext --sync pymor.md

pdf: venv
	. venv/bin/activate && jupyter nbconvert --to slides your_talk.ipynb --post serve

# this way we only re-run install if requirements change
venv/setup_by_make: requirements.txt
	test -d venv || python3 -m venv venv
	. venv/bin/activate && python3 -m pip install -q -r requirements-dev.txt
	touch venv/setup_by_make

venv: venv/setup_by_make

qr_%.png: venv
	. venv/bin/activate && python render_qr.py

clean:
	rm qr_*png
	rm pymor.ipynb
