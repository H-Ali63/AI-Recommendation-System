#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════
#  deploy.sh — RecommendAI VPS Deployment Script
#
#  Usage:
#    chmod +x deploy.sh
#    ./deploy.sh [--domain your-domain.com] [--ssl]
#
#  Tested on: Ubuntu 22.04 LTS / Debian 12
# ════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()    { echo -e "\n${BOLD}${BLUE}▶ $*${NC}"; }

# ── Parse arguments ───────────────────────────────────────────
DOMAIN=""
ENABLE_SSL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --domain) DOMAIN="$2"; shift 2 ;;
    --ssl)    ENABLE_SSL=true; shift ;;
    *) warn "Unknown argument: $1"; shift ;;
  esac
done

# ════════════════════════════════════════════════════════════════
step "1. System update & dependencies"
# ════════════════════════════════════════════════════════════════
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  curl git ca-certificates gnupg lsb-release \
  ufw fail2ban

success "System packages installed"

# ════════════════════════════════════════════════════════════════
step "2. Install Docker & Docker Compose"
# ════════════════════════════════════════════════════════════════
if ! command -v docker &>/dev/null; then
  info "Installing Docker…"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  success "Docker installed"
else
  success "Docker already installed ($(docker --version))"
fi

if ! docker compose version &>/dev/null; then
  info "Installing Docker Compose plugin…"
  sudo apt-get install -y docker-compose-plugin
  success "Docker Compose installed"
else
  success "Docker Compose already available"
fi

# ════════════════════════════════════════════════════════════════
step "3. Configure firewall (UFW)"
# ════════════════════════════════════════════════════════════════
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp   comment 'HTTP'
sudo ufw allow 443/tcp  comment 'HTTPS'
sudo ufw --force enable
success "Firewall configured"

# ════════════════════════════════════════════════════════════════
step "4. Configure fail2ban"
# ════════════════════════════════════════════════════════════════
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
success "fail2ban active"

# ════════════════════════════════════════════════════════════════
step "5. Set up .env file"
# ════════════════════════════════════════════════════════════════
if [[ ! -f .env ]]; then
  cp .env.example .env
  # Generate a secure secret key
  SECRET_KEY=$(openssl rand -hex 32)
  sed -i "s/your-secret-key-change-in-production-use-openssl-rand-hex-32/${SECRET_KEY}/" .env
  sed -i "s/APP_ENV=development/APP_ENV=production/" .env

  if [[ -n "$DOMAIN" ]]; then
    sed -i "s|ALLOWED_ORIGINS=http://localhost:8501,http://localhost:3000|ALLOWED_ORIGINS=https://${DOMAIN}|" .env
    sed -i "s|BACKEND_URL=http://backend:8000|BACKEND_URL=https://${DOMAIN}/api/v1|" .env
  fi

  warn ".env created — review and update MONGO_URI, REDIS passwords, etc."
else
  info ".env already exists — skipping"
fi

# ════════════════════════════════════════════════════════════════
step "6. SSL certificate (Let's Encrypt)"
# ════════════════════════════════════════════════════════════════
if $ENABLE_SSL && [[ -n "$DOMAIN" ]]; then
  if ! command -v certbot &>/dev/null; then
    sudo apt-get install -y certbot
  fi

  mkdir -p nginx/certs

  sudo certbot certonly --standalone \
    --non-interactive \
    --agree-tos \
    --register-unsafely-without-email \
    -d "$DOMAIN" \
    --pre-hook "docker compose stop nginx" \
    --post-hook "docker compose start nginx"

  sudo cp /etc/letsencrypt/live/"$DOMAIN"/fullchain.pem nginx/certs/
  sudo cp /etc/letsencrypt/live/"$DOMAIN"/privkey.pem   nginx/certs/
  sudo chown "$USER":"$USER" nginx/certs/*

  # Uncomment HTTPS server block in nginx.conf
  sed -i "s|# server {|server {|g; s|#     listen 443|    listen 443|g" nginx/nginx.conf
  sed -i "s|# return 301|return 301|g" nginx/nginx.conf
  sed -i "s|your-domain.com|${DOMAIN}|g" nginx/nginx.conf

  # Auto-renew cron
  (crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet --post-hook 'docker compose -f $(pwd)/docker-compose.yml restart nginx'") | crontab -
  success "SSL configured for $DOMAIN"
else
  info "Skipping SSL (use --domain your-domain.com --ssl to enable)"
fi

# ════════════════════════════════════════════════════════════════
step "7. Build & start services"
# ════════════════════════════════════════════════════════════════
docker compose build --no-cache
docker compose up -d

info "Waiting for services to become healthy…"
sleep 15

# ── Health checks ─────────────────────────────────────────────
check_service() {
  local name=$1 url=$2 retries=10 delay=8
  for i in $(seq 1 $retries); do
    if curl -sf "$url" &>/dev/null; then
      success "$name is healthy"
      return 0
    fi
    info "Waiting for $name ($i/$retries)…"
    sleep $delay
  done
  warn "$name did not become healthy in time — check: docker compose logs $name"
}

check_service "Backend API"  "http://localhost:80/health"
check_service "Frontend UI"  "http://localhost:80/_stcore/health"

# ════════════════════════════════════════════════════════════════
step "8. Set up systemd service (auto-start on reboot)"
# ════════════════════════════════════════════════════════════════
WORKDIR=$(pwd)
sudo tee /etc/systemd/system/recommendai.service > /dev/null <<EOF
[Unit]
Description=RecommendAI Docker Compose Stack
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${WORKDIR}
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable recommendai
success "systemd service enabled (auto-starts on reboot)"

# ════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  RecommendAI deployed successfully! 🚀 ${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
echo ""
if [[ -n "$DOMAIN" ]]; then
  echo -e "  Frontend:  ${CYAN}https://$DOMAIN${NC}"
  echo -e "  API Docs:  ${CYAN}https://$DOMAIN/docs${NC}"
  echo -e "  Health:    ${CYAN}https://$DOMAIN/health${NC}"
else
  echo -e "  Frontend:  ${CYAN}http://<YOUR-SERVER-IP>${NC}"
  echo -e "  API Docs:  ${CYAN}http://<YOUR-SERVER-IP>/docs${NC}"
  echo -e "  Health:    ${CYAN}http://<YOUR-SERVER-IP>/health${NC}"
fi
echo ""
echo -e "  Useful commands:"
echo -e "    ${YELLOW}docker compose logs -f backend${NC}   # Stream backend logs"
echo -e "    ${YELLOW}docker compose logs -f frontend${NC}  # Stream frontend logs"
echo -e "    ${YELLOW}docker compose ps${NC}                 # Check service status"
echo -e "    ${YELLOW}docker compose restart backend${NC}   # Restart backend"
echo -e "    ${YELLOW}docker compose down && docker compose up -d${NC}  # Full restart"
echo ""
