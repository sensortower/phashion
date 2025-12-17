# Build stage
FROM ubuntu:24.04 AS builder

WORKDIR /build

# Install build dependencies
# You can add additional packages here as needed
RUN apt-get update && apt-get install -y \
    build-essential \
    ruby-dev \
    git \
    ruby-bundler \
    libjpeg-dev \
    libpng-dev \
    libsqlite3-dev \
    imagemagick \
    tar \
    && rm -rf /var/lib/apt/lists/*


# Copy gem files
COPY . .

# Install gem dependencies (including development dependencies for rake-compiler)
RUN bundle install

# Build the gem
RUN rake compile

RUN rake test

RUN rake native gem

# Run stage
FROM ubuntu:24.04 AS runner

# Install only Ruby runtime
 RUN apt-get update && apt-get install -y \
    ruby \
    ruby-dev \
    libjpeg-dev \
    libpng-dev \
    imagemagick \
     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built gem from builder stage
COPY --from=builder /build/pkg/*.gem .

#RUN gem install --local /app/$(ruby -e 'puts "#{Gem::Platform.local.cpu}-#{Gem::Platform.local.os}-#{Gem::Platform.local.version}"').gem
RUN PLATFORM=$(ruby -e 'puts "#{Gem::Platform.local.cpu}-#{Gem::Platform.local.os}-#{Gem::Platform.local.version}"') && \
    GEMFILE=$(find . -name "*$PLATFORM.gem") && \
    gem install --local $GEMFILE

COPY run_phashion.rb run_phashion.rb

# Set entrypoint
ENTRYPOINT ["ruby", "/app/run_phashion.rb"]

