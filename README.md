# AutoGlossary

Convierte automáticamente documentos PDF en glosarios interactivos y exportables a HTML/JSON. El sistema extrae términos clave y acrónimos con sus definiciones, permitiendo filtrar, marcar favoritos y analizar glosarios con una interfaz moderna creada con Streamlit.

Desarrollado por zusldev.

![demo](docs/demo.gif)

## 🚀 Características principales

- Sube uno o varios archivos PDF y obtén glosarios extraídos de manera automática
- Detecta glosarios reales (si existen en el PDF) y los estructura instantáneamente
- Extrae términos clave y acrónimos usando técnicas de procesamiento de lenguaje
- Presenta las definiciones en contexto, incluso cuando no hay glosario formal
- Filtros avanzados por archivo, etiquetas, términos y favoritos ⭐
- Exporta resultados en formato HTML o JSON listo para compartir
- Interfaz interactiva y profesional con Streamlit
- Soporte para español e inglés (más idiomas fácilmente extensibles)
- Historial en sesión, selección de términos favoritos y ajustes personalizables

## 🖥️ Capturas de pantalla

<!-- Si agregas imágenes reales del UI, colócalas aquí (opcional) -->

## ⚡ Instalación rápida

1. Clona el repositorio:
   ```bash
   git clone https://github.com/zusldev/auto_glossary.git
   cd auto_glossary
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Uso

Ejecuta la aplicación localmente:

```bash
streamlit run app_glosario.py
```

Sube tus archivos PDF o prueba con `demo.pdf` incluido en el proyecto. Ajusta las opciones en la barra lateral de la interfaz para personalizar la extracción y análisis.

## Requisitos

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- NumPy, Pandas, scikit-learn
- PyMuPDF (fitz)

Instala todo usando el archivo requirements.txt incluido.

## Estructura del Proyecto

```
├── app_glosario.py         # Lógica principal y UI en Streamlit
├── demo.pdf               # PDF de ejemplo para pruebas rápidas
├── requirements.txt       # Dependencias Python
├── LICENSE                # Licencia del proyecto (MIT)
├── .gitignore             # Exclusiones para git
```

## Desarrollo y contribución

1. Haz un fork y crea una nueva rama para tus cambios
2. Abre un Pull Request con tu propuesta
3. Abre issues para reporte de bugs o sugerencias

¡Se reciben aportes de la comunidad!

## Licencia

Este proyecto está licenciado bajo términos MIT. Consulta el archivo LICENSE para más detalles.

## Autor

Creado y mantenido por zusldev

## Notas

- Si tienes problemas con PDFs particulares, puedes abrir un issue para soporte o mejora.
- El proyecto nunca sube tus archivos PDF a la nube: el análisis se ejecuta localmente.
- El archivo demo.pdf es solo un ejemplo de uso y puede ser reemplazado por cualquier PDF propio.

---

> Proyecto open-source profesional para la extracción rápida de glosarios desde PDF utilizando Python y Streamlit.
