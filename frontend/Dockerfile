FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy rest of the code
COPY . .

# Build the SvelteKit app
RUN npm run build

# Expose the port used by SvelteKit
EXPOSE 3000

# Run the app
CMD ["node", "build"]
