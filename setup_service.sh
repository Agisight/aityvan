#!/bin/bash
cat > /etc/systemd/system/translator.service << 'EOF'
[Unit]
Description=Tyvan Translator
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/translator/app/app-content
ExecStart=/opt/translator/app/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
MemoryMax=8G
MemoryHigh=7G

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable translator
systemctl start translator
systemctl status translator
