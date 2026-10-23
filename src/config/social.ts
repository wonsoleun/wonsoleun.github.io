import type { SocialLink } from "../types";

export const SOCIALS: SocialLink[] = [
    {
        name: "Github",
        href: "https://github.com/wonsoleun",
        linkTitle: `Follow Soleun Won on Github`,
        isActive: true,
    },
    {
        name: "Mail",
        href: "mailto:ky9082ann@gmai.com",
        linkTitle: `Send an email to Soleun Won`,
        isActive: true,
    },
    {
        name: "Google Scholar",
        href: "https://scholar.google.com/citations?user=shannon",
        linkTitle: `Claude Shannon on Google Scholar`,
        isActive: false,
    },
    {
        name: "ORCID",
        href: "https://orcid.org/0000-0002-1825-0097",
        linkTitle: `Claude Shannon on ORCID`,
        isActive: false,
    },
    {
        name: "LinkedIn",
        href: "https://www.linkedin.com/in/soleun-won-a042703a3/",
        linkTitle: `Soleun Won on LinkedIn`,
        isActive: true, // Assuming Claude doesn't have a LinkedIn profile
    },
];

export const SOCIAL_ICONS: Record<string, string> = {
    Github: "Github",
    Mail: "Mail",
    Linkedin: "LinkedIn",
    "Google Scholar": "GoogleScholar",
    ORCID: "ORCID",
    RSS: "RSS",
};