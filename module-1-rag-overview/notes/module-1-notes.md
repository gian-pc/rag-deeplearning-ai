# Módulo 1 — RAG Overview

## ¿Qué es RAG?
RAG (Retrieval Augmented Generation) combina un LLM con una base de conocimiento privada.
El retriever busca documentos relevantes y los inyecta en el prompt antes de enviárselo al modelo.

## ¿Por qué necesitamos RAG?
Los LLMs solo saben lo que aprendieron en su entrenamiento.
No tienen acceso a datos privados, recientes o muy específicos.
RAG resuelve esto sin necesidad de re-entrenar el modelo.

## Conceptos clave

### Alucinación
Cuando el LLM no tiene los datos, inventa respuestas que suenan convincentes pero son incorrectas.
Ejemplo: preguntarle por las casas de tu inmobiliaria sin darle los datos.

### Prompt aumentado
Un prompt que contiene la pregunta del usuario + datos relevantes inyectados.
El modelo usa esos datos para responder correctamente.

### Tokens
Unidad de texto que procesa el modelo. 1 token ≈ 1 palabra.
Cada token tiene un costo — por eso limitamos `max_tokens`.

### Roles en una conversación
- `system` → instrucciones de comportamiento para el modelo
- `user` → tus mensajes
- `assistant` → respuestas del modelo

## Flujo básico de RAG
1. Usuario hace una pregunta
2. Retriever busca documentos relevantes
3. Se arma un prompt aumentado: datos + pregunta
4. El LLM responde basándose en los datos proporcionados

## Labs completados
- [x] Lab 2 — Llamadas al LLM y prompts aumentados
