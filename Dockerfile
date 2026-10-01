FROM python:3.11-slim

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py module1.py module2_telemetry_sim.py \
     module3_orchestrator.py module4_quantization_sim.py ./

# Copy the HTML template
COPY templates/ ./templates/

# Give the app user ownership of the workdir (needed to write the forecast CSV)
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 5000

CMD ["python", "app.py"]