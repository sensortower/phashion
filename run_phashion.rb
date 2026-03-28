require 'phashion'

puts Phashion::Image.new(ENV['IMAGE_PATH']).fingerprint
