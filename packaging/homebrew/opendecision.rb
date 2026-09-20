# Homebrew FORMULA template (CLI & daemon). Source of truth; the release workflow
# renders the @@...@@ tokens from release assets and pushes to whit3rabbit/homebrew-tap.
# Install: brew install whit3rabbit/tap/opendecision
class Opendecision < Formula
  desc "Independent Rust decision-inference engine that speaks the Jev protocol"
  homepage "https://github.com/whit3rabbit/opendecision"
  version "@@VERSION@@"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/whit3rabbit/opendecision/releases/download/v#{version}/opendecision-#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "@@SHA_ARM_TGZ@@"
    end
    on_intel do
      url "https://github.com/whit3rabbit/opendecision/releases/download/v#{version}/opendecision-#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "@@SHA_INTEL_TGZ@@"
    end
  end

  on_linux do
    url "https://github.com/whit3rabbit/opendecision/releases/download/v#{version}/opendecision-#{version}-x86_64-unknown-linux-musl.tar.gz"
    sha256 "@@SHA_LINUX_TGZ@@"
  end

  def install
    bin.install "opendecision"
    bin.install "opendecisiond"
  end

  test do
    system bin/"opendecision", "version"
  end
end
