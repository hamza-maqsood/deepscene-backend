# deepscene-backend

<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="./app/resources/Logo.png" alt="Project logo"></a>
</p>

<h3 align="center">DeepScene</h3>

---

<p align="center"> An AI capable of generating a Scene from a Script.
    <br> 
</p>

## 🧐 About (Optional)
This is a final year research project of students from NUCES ISB.

## 🏁 Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing
Please follow the following steps to install:

Run the following commands
```
pip install -r requirements.txt
```

## Structure
The project is structured in the following manner
```
.
├── app
│   ├── __init__.py
│   ├── main.py
│   └── routers
│   │   ├── __init__.py
│   │   ├── common_sense.py
│   │   └── entit.py
│   │   └── nlp.py
```

## 🎈 Usage
Launch the server using the following command:
```
uvicorn main:app --reload
```
How to run
    
    uvicorn app.main:app --reload --port=9000

## 🚀 Deployment
Add additional notes about how to deploy this on a live system.

## 📝 Docs
- [Swagger](/doc) - Main Documentation
- [ReDoc](/redoc) - Alternate Documentation

## ✍️ Authors
- [hamza-maqsood](https://github.com/hamza-maqsood)
- [@zohairhadi](https://github.com/zohairhadi)
- [@mannan]((https://github.com/mannan))

## 🎉 Acknowledgements
- Good enough