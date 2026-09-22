import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav className="navbar">

      <Link to="/" className="logo">
        DeceptiUI
      </Link>

      <div className="nav-links">

        <Link
          to="/"
          className="nav-link"
        >
          Home
        </Link>

        <a
          href="/#how-it-works"
          className="nav-link"
        >
          How It Works
        </a>

        <a
          href="/#research"
          className="nav-link"
        >
          Research
        </a>

        <Link
          to="/analyze"
          className="nav-link analyze-nav"
        >
          Analyze Screenshot
        </Link>

      </div>

    </nav>
  );
}

export default Navbar;