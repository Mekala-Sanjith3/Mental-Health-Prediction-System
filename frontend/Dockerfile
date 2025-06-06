# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (including serve for production)
RUN npm ci

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage - using serve instead of nginx for Railway compatibility
FROM node:18-alpine

WORKDIR /app

# Install serve globally
RUN npm install -g serve

# Copy built app
COPY --from=build /app/build ./build

# Copy package.json for serve configuration
COPY --from=build /app/package.json ./

# Railway uses PORT env variable
ENV PORT=3000
EXPOSE $PORT

# Start the app with serve
CMD serve -s build -l $PORT 