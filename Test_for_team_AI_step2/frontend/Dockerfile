# Build
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package.json ./package.json
COPY frontend/vite.config.js ./vite.config.js
COPY frontend/index.html ./index.html
COPY frontend/src ./src
RUN npm install && npm run build
# Serve via nginx and proxy /api to backend
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx","-g","daemon off;"]
