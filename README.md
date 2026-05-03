## Entorno Python y kernel de notebooks

Desde la raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name grupo3-cobranzas --display-name "Grupo 3 Cobranzas"
```

Despues de ejecutar esos comandos, selecciona el kernel **Grupo 3 Cobranzas** en VS Code o Jupyter para correr los notebooks del proyecto.

Notas:

- `.venv/` no se sube a Git; cada computadora debe reconstruirlo con `requirements.txt`.
- El kernel queda registrado localmente en la maquina del usuario.
- Todos los notebooks del repositorio deben usar el mismo kernel para evitar diferencias entre fases.
