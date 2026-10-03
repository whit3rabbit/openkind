# Homebrew FORMULA template (CLI & daemon). Source of truth; the release workflow
# renders the @@...@@ tokens from release assets and pushes to whit3rabbit/homebrew-tap.
# Install: brew install whit3rabbit/tap/openkind
require "json"

class Openkind < Formula
  desc "Independent Rust decision-inference engine that speaks the Jev protocol"
  homepage "https://github.com/whit3rabbit/openkind"
  version "@@VERSION@@"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/whit3rabbit/openkind/releases/download/v#{version}/openkind-#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "@@SHA_ARM_TGZ@@"
    end
    on_intel do
      url "https://github.com/whit3rabbit/openkind/releases/download/v#{version}/openkind-#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "@@SHA_INTEL_TGZ@@"
    end
  end

  on_linux do
    url "https://github.com/whit3rabbit/openkind/releases/download/v#{version}/openkind-#{version}-x86_64-unknown-linux-gnu.tar.gz"
    sha256 "@@SHA_LINUX_TGZ@@"
  end

  def install
    bin.install "openkind"
    bin.install "openkindd"
    bin.install "mlx.metallib" if File.exist?("mlx.metallib")
  end

  test do
    system bin/"openkind", "version"
    report = JSON.parse(shell_output("#{bin}/openkind doctor --json"))
    assert_equal true, report.fetch("backends").find { |b| b.fetch("backend") == "native-cpu" }.fetch("ready")
  end
end
